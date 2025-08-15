
"""
Mobile App Backend with Authentication and Database Integration
Flask API for mobile app with user management
"""
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import sqlite3
import hashlib
import jwt
import datetime
from functools import wraps
import os
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app requests

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-key')
DATABASE = 'mobile_app.db'

def init_database():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            full_name VARCHAR(100),
            phone VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            profile_picture VARCHAR(255)
        )
    ''')
    
    # User sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            device_info VARCHAR(255),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # User preferences table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            favorite_stocks TEXT,
            notification_settings TEXT,
            theme VARCHAR(20) DEFAULT 'light',
            language VARCHAR(10) DEFAULT 'en',
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Portfolio table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            stock_symbol VARCHAR(20) NOT NULL,
            quantity INTEGER NOT NULL,
            purchase_price DECIMAL(10,2) NOT NULL,
            purchase_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    return hashlib.sha256(password.encode()).hexdigest() == hashed

def generate_token(user_id):
    """Generate JWT token for user"""
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, app.config['JWT_SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    """Decorator to require valid token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            user_id = verify_token(token)
            if not user_id:
                return jsonify({'error': 'Token is invalid'}), 401
        except:
            return jsonify({'error': 'Token is invalid'}), 401
        
        return f(user_id, *args, **kwargs)
    return decorated

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    return True, "Password is valid"

@app.route('/api/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['username', 'email', 'password', 'full_name']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        full_name = data['full_name'].strip()
        phone = data.get('phone', '').strip()
        
        # Validate email
        if not validate_email(email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Validate password
        is_valid, message = validate_password(password)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Check if user already exists
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if cursor.fetchone():
            conn.close()
            return jsonify({'error': 'Username or email already exists'}), 409
        
        # Create new user
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, phone)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, password_hash, full_name, phone))
        
        user_id = cursor.lastrowid
        
        # Create user preferences
        cursor.execute('''
            INSERT INTO user_preferences (user_id)
            VALUES (?)
        ''', (user_id,))
        
        conn.commit()
        conn.close()
        
        # Generate token
        token = generate_token(user_id)
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': {
                'id': user_id,
                'username': username,
                'email': email,
                'full_name': full_name
            }
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        
        if not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Username and password are required'}), 400
        
        username = data['username'].strip()
        password = data['password']
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Check if login is by email or username
        if validate_email(username):
            cursor.execute('''
                SELECT id, username, email, password_hash, full_name, is_active
                FROM users WHERE email = ?
            ''', (username.lower(),))
        else:
            cursor.execute('''
                SELECT id, username, email, password_hash, full_name, is_active
                FROM users WHERE username = ?
            ''', (username,))
        
        user = cursor.fetchone()
        conn.close()
        
        if not user or not verify_password(password, user[3]):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not user[5]:  # is_active
            return jsonify({'error': 'Account is deactivated'}), 401
        
        # Generate token
        token = generate_token(user[0])
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'full_name': user[4]
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile', methods=['GET'])
@token_required
def get_profile(user_id):
    """Get user profile"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT u.id, u.username, u.email, u.full_name, u.phone, u.created_at,
                   p.favorite_stocks, p.theme, p.language
            FROM users u
            LEFT JOIN user_preferences p ON u.id = p.user_id
            WHERE u.id = ?
        ''', (user_id,))
        
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user': {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'full_name': user[3],
                'phone': user[4],
                'created_at': user[5],
                'favorite_stocks': user[6].split(',') if user[6] else [],
                'theme': user[7],
                'language': user[8]
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile', methods=['PUT'])
@token_required
def update_profile(user_id):
    """Update user profile"""
    try:
        data = request.get_json()
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Update user basic info
        if 'full_name' in data:
            cursor.execute('UPDATE users SET full_name = ? WHERE id = ?', 
                         (data['full_name'], user_id))
        
        if 'phone' in data:
            cursor.execute('UPDATE users SET phone = ? WHERE id = ?', 
                         (data['phone'], user_id))
        
        # Update preferences
        if 'favorite_stocks' in data:
            favorite_stocks = ','.join(data['favorite_stocks']) if data['favorite_stocks'] else ''
            cursor.execute('UPDATE user_preferences SET favorite_stocks = ? WHERE user_id = ?',
                         (favorite_stocks, user_id))
        
        if 'theme' in data:
            cursor.execute('UPDATE user_preferences SET theme = ? WHERE user_id = ?',
                         (data['theme'], user_id))
        
        if 'language' in data:
            cursor.execute('UPDATE user_preferences SET language = ? WHERE user_id = ?',
                         (data['language'], user_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Profile updated successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio', methods=['GET'])
@token_required
def get_portfolio(user_id):
    """Get user portfolio"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT stock_symbol, quantity, purchase_price, purchase_date
            FROM portfolios
            WHERE user_id = ?
            ORDER BY purchase_date DESC
        ''', (user_id,))
        
        portfolio = cursor.fetchall()
        conn.close()
        
        portfolio_data = []
        for item in portfolio:
            portfolio_data.append({
                'stock_symbol': item[0],
                'quantity': item[1],
                'purchase_price': float(item[2]),
                'purchase_date': item[3]
            })
        
        return jsonify({'portfolio': portfolio_data}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio', methods=['POST'])
@token_required
def add_to_portfolio(user_id):
    """Add stock to portfolio"""
    try:
        data = request.get_json()
        
        required_fields = ['stock_symbol', 'quantity', 'purchase_price']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO portfolios (user_id, stock_symbol, quantity, purchase_price)
            VALUES (?, ?, ?, ?)
        ''', (user_id, data['stock_symbol'], data['quantity'], data['purchase_price']))
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Stock added to portfolio successfully'}), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
@token_required
def logout(user_id):
    """Logout user"""
    try:
        # In a more complex system, you would invalidate the token
        # For now, we'll just return success
        return jsonify({'message': 'Logout successful'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/change-password', methods=['POST'])
@token_required
def change_password(user_id):
    """Change user password"""
    try:
        data = request.get_json()
        
        if not data.get('current_password') or not data.get('new_password'):
            return jsonify({'error': 'Current password and new password are required'}), 400
        
        # Validate new password
        is_valid, message = validate_password(data['new_password'])
        if not is_valid:
            return jsonify({'error': message}), 400
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Verify current password
        cursor.execute('SELECT password_hash FROM users WHERE id = ?', (user_id,))
        current_hash = cursor.fetchone()[0]
        
        if not verify_password(data['current_password'], current_hash):
            conn.close()
            return jsonify({'error': 'Current password is incorrect'}), 401
        
        # Update password
        new_hash = hash_password(data['new_password'])
        cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Password changed successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.utcnow().isoformat()
    }), 200

@app.route('/api/stocks/popular', methods=['GET'])
def get_popular_stocks():
    """Get popular Indian stocks"""
    from config.settings import POPULAR_STOCKS
    return jsonify({'stocks': POPULAR_STOCKS}), 200

if __name__ == '__main__':
    init_database()
    app.run(host='0.0.0.0', port=5000, debug=True)
