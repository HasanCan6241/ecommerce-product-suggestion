from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
import secrets
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import re
from flask_sqlalchemy import SQLAlchemy
from math import ceil


# Flask uygulamasını başlat

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # 16 byte uzunluğunda güvenli bir anahtar

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ecommerce.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Product model
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.String(50), unique=True, nullable=False)
    product_name = db.Column(db.String(150), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    discounted_price = db.Column(db.Float, nullable=False)
    actual_price = db.Column(db.Float, nullable=False)
    discount_percentage = db.Column(db.Float, nullable=False)
    rating = db.Column(db.Float, nullable=False)
    rating_count = db.Column(db.Integer, nullable=False)
    about_product = db.Column(db.Text, nullable=False)
    img_link = db.Column(db.String(300), nullable=False)
    product_link = db.Column(db.String(300), nullable=False)

# Cart model
class Cart(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_id = db.Column(db.String(50), db.ForeignKey('product.product_id'), nullable=False)
    product = db.relationship('Product', backref='cart_items')

"""# Function to reset and load data only once
def reset_and_load_db(app, db, Product):
    with app.app_context():
        # Drop and create all tables
        db.drop_all()  # Drop all tables
        db.create_all()  # Create tables

        # Proceed to add data only if the product table is empty
        if not Product.query.first():  # If no products exist
            try:
                # Read the cleaned dataset
                df = pd.read_csv("cleaned_amazon.csv")

                # Create a set to track existing product_ids
                existing_product_ids = set()

                # Create list for bulk insertion
                products_to_add = []

                for _, row in df.iterrows():
                    product_id = str(row['product_id'])

                    # Skip if product_id already exists
                    if product_id in existing_product_ids:
                        continue

                    existing_product_ids.add(product_id)

                    product = Product(
                        product_id=product_id,
                        product_name=row['product_name'],
                        category=row['category'],
                        discounted_price=float(row['discounted_price']),
                        actual_price=float(row['actual_price']),
                        discount_percentage=float(row['discount_percentage']),
                        rating=float(row['rating']),
                        rating_count=int(row['rating_count']),
                        about_product=row['about_product'],
                        img_link=row['img_link'],
                        product_link=row['product_link']
                    )
                    products_to_add.append(product)

                # Perform bulk insertion
                db.session.bulk_save_objects(products_to_add)
                db.session.commit()
                print(f"{len(products_to_add)} products successfully loaded into database.")

            except Exception as e:
                db.session.rollback()
                print(f"Error occurred: {str(e)}")
                raise"""

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Öncelikle modify_url fonksiyonunu tanımlıyoruz
def modify_url(url):
    # Eğer URL zaten doğru formattaysa, olduğu gibi döndür
    if re.match(r"^https://m\.media-amazon\.com/images/I/.+\.jpg$", url):
        return url

    # Sabit başlangıç kısmı
    fixed_part = "https://m.media-amazon.com/images/I/"

    # URL'deki 'images/W/WEBP_' kısmını düzenle
    pattern = re.compile(r"https://m\.media-amazon\.com/images/[^/]+/WEBP_([^/]+)/images/I/([^/]+)\.jpg")
    match = pattern.search(url)

    if match:
        # Elde edilen kısmı sabit başlangıç kısmına ekleyin
        modified_url = fixed_part + match.group(2) + ".jpg"
        return modified_url
    else:
        return "Geçerli URL bulunamadı!"

# Load and preprocess dataset
def preprocess_data():
    df = pd.read_csv("amazon.csv")
    df.dropna(subset=['rating_count', 'rating', 'about_product'], inplace=True)
    # Çalışmayan linklerin düzenlenmesi
    df['img_link'] = df['img_link'].apply(modify_url)
    df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False).astype(float)
    df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False).astype(float)
    df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '', regex=False).astype(float) / 100
    df = df[df['rating'].apply(lambda x: '|' not in str(x))]
    df['rating'] = df['rating'].astype(str).str.replace(',', '', regex=False).astype(float)
    df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '', regex=False).astype(float)
    df['rating_weighted'] = df['rating'] * df['rating_count']
    df['sub_category'] = df['category'].astype(str).str.split('|').str[-1]
    df['main_category'] = df['category'].astype(str).str.split('|').str[0]
    return df

product_data = preprocess_data()
product_data.to_csv("cleaned_amazon.csv", index=False)


# Model sınıfını güncelleyin
class RecommendationModel:
    def __init__(self, tfidf_matrix, product_data):
        self.tfidf_matrix = tfidf_matrix
        self.product_data = product_data

    def recommend(self, product_ids): #ürün ıd alıyor
        recommended_products = []

        # Sepetteki her bir ürün için öneri yap
        for product_id in product_ids:
            indices = self.product_data[self.product_data['product_id'] == product_id].index
            if len(indices) == 0:
                print(f"No matching product found for ID: {product_id}")
                continue

            # İlgili ürünün kategorisini al ve hiyerarşiyi çöz
            product_category = self.product_data.loc[indices[0], 'category']
            category_hierarchy = product_category.split('|')  # Kategoriyi hiyerarşi olarak ayır

            # Aynı kategori veya alt kategori
            matching_products = self._find_matching_products(category_hierarchy)

            if matching_products.empty:  # Eğer eşleşen ürün yoksa, öneri yapılmaz
                print(f"No matching products found in category for product ID: {product_id}")
                continue

            # TF-IDF matrisinden aynı kategoriye ait ürünlerin vektörlerini seç
            matching_indices = matching_products.index

            # Burada, matching_indices'in boyutunun tfidf_matrix'in boyutlarıyla uyumlu olup olmadığını kontrol et
            if any(idx >= self.tfidf_matrix.shape[0] for idx in matching_indices):
                print(f"Warning: Some indices are out of bounds for the TF-IDF matrix.")
                continue

            matching_tfidf = self.tfidf_matrix[matching_indices]

            # Cosine similarity hesaplama
            cosine_sim_user = cosine_similarity(self.tfidf_matrix[indices], matching_tfidf)
            scores = cosine_sim_user.mean(axis=0)

            # En benzer ürünleri bul
            top_indices = scores.argsort()[::-1]  # Skorları büyükten küçüğe sırala
            for idx in top_indices:
                recommended_index = matching_indices[idx]  # Orijinal veri kümesindeki indeks
                if recommended_index == indices[0]:  # Aynı ürünü atla
                    continue
                if self.product_data.iloc[recommended_index]['product_id'] in product_ids:  # Sepetteki ürünleri atla
                    continue
                recommended_products.append(self.product_data.iloc[recommended_index].to_dict())
                if len(recommended_products) >= 2:  # Her ürün için 2 öneri sınırı
                    break

        # Önerilen ürünleri eşsiz hale getirin
        unique_recommendations = {prod['product_id']: prod for prod in recommended_products}
        return list(unique_recommendations.values())

    def _find_matching_products(self, category_hierarchy):
        """Aynı kategori veya alt kategorilerdeki ürünleri bul."""
        for depth in range(len(category_hierarchy), 0, -1):
            partial_category = '|'.join(category_hierarchy[:depth])
            matching_products = self.product_data[
                self.product_data['category'].str.startswith(partial_category)
            ]
            if not matching_products.empty:
                return matching_products
        return self.product_data.iloc[0:0]  # Boş bir DataFrame döndür

# Load recommendation model
model_path = "recommendation_model.pkl"
try:
    recommendation_model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading recommendation model: {e}")
    recommendation_model = None


def recommend_products(cart_products):
    return recommendation_model.recommend(cart_products)


@app.route('/')
def home():
    # Kullanıcının filtreleme parametrelerini al
    product_name = request.args.get('product_name', '')
    category = request.args.get('category', '')
    min_price = request.args.get('min_price', type=float)
    max_price = request.args.get('max_price', type=float)
    min_rating = request.args.get('min_rating', type=float)

    page = request.args.get('page', 1, type=int)
    per_page = 20  # Her sayfada gösterilecek ürün sayısı

    # Ürün kategorilerini veritabanından çek
    categories = db.session.query(Product.category.distinct()).all()
    categories = [category[0] for category in categories]  # Kategorileri listeye dönüştür

    # Ürünleri filtrele
    query = Product.query

    if product_name:
        query = query.filter(Product.product_name.ilike(f"%{product_name}%"))
    if category:
        query = query.filter(Product.category == category)
    if min_price:
        query = query.filter(Product.discounted_price >= min_price)
    if max_price:
        query = query.filter(Product.discounted_price <= max_price)
    if min_rating is not None:
        query = query.filter(Product.rating >= min_rating)

    # Toplam ürün sayısını al
    total_products = query.count()

    # Sayfa sayısını hesapla
    total_pages = ceil(total_products / per_page)

    # Ürünleri sayfalı şekilde al
    products = query.order_by(Product.id).paginate(page=page, per_page=per_page, error_out=False)

    pagination = {
        'page': page,
        'total_pages': total_pages,
        'has_next': page < total_pages,
        'has_prev': page > 1,
        'pages': range(1, total_pages + 1)
    }

    return render_template('home.html', products=products.items, pagination=pagination, categories=categories)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        if User.query.filter_by(email=email).first():
            flash('Email already exists.')
            return redirect(url_for('register'))

        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))

        flash('Invalid credentials.')
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/product/<int:product_id>')
def product_detail(product_id):
    product = Product.query.get_or_404(product_id)
    return render_template('product_detail.html', product=product)


# Cart route - route'u güncelleyin
# Cart route'u güncelleyin
# Cart route'u güncelleyin
@app.route('/cart')
@login_required
def cart():
    # Kullanıcının sepetindeki ürünleri al
    cart_items = Cart.query.filter_by(user_id=current_user.id).all()

    # Sepetteki ürün ID'lerini topla
    cart_product_ids = [item.product_id for item in cart_items]

    # Önerilen ürünleri hazırlayın
    recommended = []
    if cart_product_ids:
        # Her ürün için önerileri ayrı ayrı saklayın
        product_recommendations = {
            product_id: recommend_products([product_id]) for product_id in cart_product_ids
        }

        # Tüm önerileri birleştirin
        for product_id, recs in product_recommendations.items():
            for product in recs:
                if 'img_link' in product:  # img_link kontrolü
                    product['img_link'] = modify_url(product['img_link'])
                recommended.append(product)

    # Önerilen ürünlerden tekrar edenleri kaldırın
    unique_recommendations = {prod['product_id']: prod for prod in recommended}
    recommended = list(unique_recommendations.values())

    # Toplam tutarı hesapla
    total = sum(item.product.discounted_price for item in cart_items)

    return render_template('cart.html',
                           cart_items=cart_items,
                           recommended=recommended,
                           total=total,
                           cart_product_ids=cart_product_ids)


# Ürün kaldırma
@app.route('/remove_from_cart/<string:product_id>')
@login_required
def remove_from_cart(product_id):
    cart_item = Cart.query.filter_by(
        user_id=current_user.id,
        product_id=product_id
    ).first()

    if cart_item:
        db.session.delete(cart_item)
        db.session.commit()
        flash('Product removed from cart.')

    return redirect(url_for('cart'))


# Ürün ekleme
@app.route('/add_to_cart/<string:product_id>')
@login_required
def add_to_cart(product_id):
    if not Cart.query.filter_by(user_id=current_user.id, product_id=product_id).first():
        new_item = Cart(user_id=current_user.id, product_id=product_id)
        db.session.add(new_item)
        db.session.commit()
        flash('Product added to cart.')
    return redirect(url_for('home'))



if __name__ == '__main__':
    with app.app_context():
        #reset_and_load_db(app, db, Product)  # Ürünleri yükle
        app.run(debug=True)

