from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import difflib, re, pyttsx3
from textblob import TextBlob
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "secret123"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///dyslexia.db"
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# ----------------------
# DATABASE MODEL
# ----------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(200))
    name = db.Column(db.String(50))
    age = db.Column(db.Integer)
    standard = db.Column(db.String(20))
    struggles = db.Column(db.String(200))

with app.app_context():
    db.create_all()

# ----------------------
# LOAD SIMPLIFIER MODEL
# ----------------------
print("🔹 Loading simplification model...")
tokenizer = PegasusTokenizer.from_pretrained("./simplifier_model")
model = PegasusForConditionalGeneration.from_pretrained("./simplifier_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----------------------
# TEXT PROCESSING FUNCTIONS
# ----------------------
def simplify_text(text):
    """Simplify complex sentences using Pegasus."""
    if not text.strip():
        return "Please enter some text to simplify."

    try:
        batch = tokenizer([text], truncation=True, padding="longest", return_tensors="pt").to(device)
        translated = model.generate(**batch, max_length=80, num_beams=5, temperature=1.2)
        simplified_sentence = tokenizer.decode(translated[0], skip_special_tokens=True)
        return simplified_sentence
    except Exception as e:
        return f"⚠️ Error simplifying text: {str(e)}"

def reading_feedback(original, spoken):
    diff = list(difflib.ndiff(original.lower().split(), spoken.lower().split()))
    missing = [w[2:] for w in diff if w.startswith('- ')]
    extra = [w[2:] for w in diff if w.startswith('+ ')]
    feedback = []
    if missing: feedback.append(f"Missed words: {', '.join(missing)}.")
    if extra: feedback.append(f"Extra words: {', '.join(extra)}.")
    if not feedback: feedback.append("Perfect reading!")
    return " ".join(feedback)

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

# ----------------------
# ROUTES
# ----------------------
@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pw = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        new_user = User(username=uname, password=pw)
        db.session.add(new_user)
        db.session.commit()
        flash("Registered successfully! Please log in.")
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/login', methods=['POST'])
def login():
    uname = request.form['username']
    pw = request.form['password']
    user = User.query.filter_by(username=uname).first()
    if user and bcrypt.check_password_hash(user.password, pw):
        session['user'] = uname
        return redirect(url_for('dashboard'))
    else:
        flash("Invalid username or password!")
        return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully.")
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('home'))
    return render_template('dashboard.html', username=session['user'])

@app.route('/profile', methods=['GET','POST'])
def profile():
    if 'user' not in session:
        return redirect(url_for('home'))
    user = User.query.filter_by(username=session['user']).first()
    if request.method == 'POST':
        user.name = request.form['name']
        user.age = request.form['age']
        user.standard = request.form['standard']
        user.struggles = request.form['struggles']
        db.session.commit()
        flash("Profile updated successfully!")
    return render_template('profile.html', user=user)

@app.route("/tools", methods=["GET", "POST"])
def tools():
    result = None
    feedback = None
    text = None

    if request.method == "POST":
        # Simplify text
        if "simplify" in request.form:
            text = request.form["text"].strip()
            result = simplify_text(text)

        # Reading feedback
        elif "feedback" in request.form:
            expected = request.form["expected"].lower().split()
            spoken = request.form["spoken"].lower().split()
            missing = [word for word in expected if word not in spoken]
            extra = [word for word in spoken if word not in expected]

            feedback_lines = []
            feedback_lines.append("Good try! 😊 Here's what I noticed:")

            if missing:
                feedback_lines.append(f"You missed {', '.join(missing)} — do you want to review those words?")
            if extra:
                feedback_lines.append(f"You added extra words: {', '.join(extra)}.")
            if not missing and not extra:
                feedback_lines.append("Perfect reading! 🎉 Great job!")

            feedback_lines.append("Would you like to listen to the correct sentence?")
            feedback = "<br>".join(f"• {line}" for line in feedback_lines)
            text = " ".join(expected)

    return render_template("tools.html", result=result, feedback=feedback, text=text)

@app.route('/mocktest', methods=['GET','POST'])
def mocktest():
    question = "Photosynthesis is the process by which green plants use sunlight to produce food."
    score = None
    if request.method == 'POST':
        answer = request.form['answer'].lower()
        score = 10 if "plants" in answer and "sunlight" in answer else 5
    return render_template('mocktest.html', question=question, score=score)

# ----------------------
# RUN APP
# ----------------------
if __name__ == '__main__':
    app.run(debug=True)
