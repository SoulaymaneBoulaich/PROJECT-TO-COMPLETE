# 📚 Library Management System - Master Relational Databases

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Ubuntu-orange.svg)

> **Build a complete library management system and master relational databases, SQL, and database design!**

This project teaches you **everything** about relational databases through a real-world application. You'll learn proper database design, relationships, queries, and best practices.

---

## 📖 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [What You'll Learn](#-what-youll-learn)
- [Understanding Relational Databases](#-understanding-relational-databases)
- [Database Design](#-database-design)
- [Prerequisites](#-prerequisites)
- [Installation Guide](#-installation-guide)
- [Project Files](#-project-files)
- [Running the Project](#-running-the-project)
- [SQL Queries Explained](#-sql-queries-explained)
- [Advanced Features](#-advanced-features)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 What This Project Does

Imagine managing a library with:
- **Books** (thousands of them!)
- **Members** (students, faculty, etc.)
- **Authors** (one book can have multiple authors)
- **Borrowing/Returning** books
- **Late fees** calculation
- **Search** functionality
- **Reports** (most popular books, overdue items, etc.)

This system handles all of that with a **proper relational database**!

**Example interactions:**
```
📚 Welcome to City Library Management System

1. Add a new book
2. Register new member
3. Borrow a book
4. Return a book
5. Search books
6. View overdue books
7. Generate reports

What would you like to do?
```

---

## 🎓 What You'll Learn

### Database Concepts
- ✅ **Relational Database Design** - How to structure data properly
- ✅ **Primary Keys & Foreign Keys** - Connecting tables together
- ✅ **One-to-Many Relationships** - One author, many books
- ✅ **Many-to-Many Relationships** - Books with multiple authors
- ✅ **Normalization** - Organizing data to avoid redundancy
- ✅ **Indexes** - Making searches lightning fast
- ✅ **Transactions** - Ensuring data consistency
- ✅ **Constraints** - Data validation at database level

### SQL Skills
- ✅ **DDL (Data Definition)** - CREATE, ALTER, DROP tables
- ✅ **DML (Data Manipulation)** - INSERT, UPDATE, DELETE data
- ✅ **DQL (Data Query)** - SELECT with JOINs, WHERE, GROUP BY
- ✅ **Aggregate Functions** - COUNT, SUM, AVG, MAX, MIN
- ✅ **Subqueries** - Queries within queries
- ✅ **Complex JOINs** - INNER, LEFT, RIGHT, FULL OUTER
- ✅ **Views** - Virtual tables for common queries

### Real-World Skills
- ✅ **PostgreSQL** (industry-standard database)
- ✅ **SQLAlchemy ORM** - Python ↔ Database communication
- ✅ **Connection Pooling** - Efficient database connections
- ✅ **Query Optimization** - Making queries fast
- ✅ **Data Migration** - Changing database structure safely
- ✅ **Backup & Recovery** - Protecting data

---

## 🧠 Understanding Relational Databases

### What is a Relational Database?

Think of it like **multiple Excel sheets that are connected**:

**❌ Bad Way (One Big Table):**
```
| BookID | Title      | Author1 | Author2 | MemberName | BorrowDate |
|--------|------------|---------|---------|------------|------------|
| 1      | Clean Code | Martin  | NULL    | John Doe   | 2024-01-01 |
| 2      | Clean Code | Martin  | NULL    | Jane Smith | 2024-01-05 |
```
**Problems:**
- Duplicate book info
- Limited to 2 authors
- Can't find all books by an author easily

**✅ Good Way (Related Tables):**

**Books Table:**
```
| BookID | Title      | ISBN          | PublishYear |
|--------|------------|---------------|-------------|
| 1      | Clean Code | 978-0132350884| 2008        |
```

**Authors Table:**
```
| AuthorID | Name           |
|----------|----------------|
| 1        | Robert Martin  |
```

**BookAuthors Table (Junction):**
```
| BookID | AuthorID |
|--------|----------|
| 1      | 1        |
```

**Members Table:**
```
| MemberID | Name       | Email           |
|----------|------------|-----------------|
| 1        | John Doe   | john@email.com  |
```

**Transactions Table:**
```
| TransactionID | BookID | MemberID | BorrowDate | ReturnDate |
|---------------|--------|----------|------------|------------|
| 1             | 1      | 1        | 2024-01-01 | NULL       |
```

Now we can:
- ✅ Add unlimited authors to any book
- ✅ Track who borrowed what and when
- ✅ No duplicate data
- ✅ Easy to query relationships

---

## 🎨 Database Design

### Our Library Database Schema

```
┌─────────────────┐
│     AUTHORS     │
├─────────────────┤
│ author_id (PK)  │◄─────┐
│ first_name      │      │
│ last_name       │      │
│ biography       │      │
│ birth_year      │      │
└─────────────────┘      │
                         │
                         │
┌─────────────────┐      │     ┌──────────────────┐
│   BOOK_AUTHORS  │      │     │      BOOKS       │
├─────────────────┤      │     ├──────────────────┤
│ book_id (FK)    │──────┼────►│ book_id (PK)     │◄───┐
│ author_id (FK)  │──────┘     │ title            │    │
└─────────────────┘            │ isbn             │    │
                               │ publisher        │    │
                               │ publish_year     │    │
                               │ pages            │    │
                               │ category         │    │
                               │ available_copies │    │
                               │ total_copies     │    │
                               └──────────────────┘    │
                                                       │
                                                       │
┌─────────────────┐                                   │
│     MEMBERS     │                                   │
├─────────────────┤                                   │
│ member_id (PK)  │◄──────┐                          │
│ first_name      │       │                          │
│ last_name       │       │                          │
│ email           │       │                          │
│ phone           │       │                          │
│ address         │       │                          │
│ join_date       │       │                          │
│ membership_type │       │                          │
│ status          │       │                          │
└─────────────────┘       │                          │
                          │                          │
                          │                          │
┌─────────────────────────┤                          │
│    TRANSACTIONS         │                          │
├─────────────────────────┤                          │
│ transaction_id (PK)     │                          │
│ book_id (FK)            │──────────────────────────┘
│ member_id (FK)          │──────────┘
│ borrow_date             │
│ due_date                │
│ return_date             │
│ status                  │
│ fine_amount             │
└─────────────────────────┘

Legend:
PK = Primary Key (unique identifier)
FK = Foreign Key (reference to another table)
```

### Relationship Types

1. **One-to-Many**: One member can borrow many books
   - Members → Transactions (1:N)

2. **Many-to-Many**: Books can have many authors, authors can write many books
   - Books ↔ Authors (N:M) via BookAuthors junction table

3. **One-to-One**: Each transaction relates to one book at a time
   - Transactions → Books (N:1)

---

## 📋 Prerequisites

### What You Need to Know
- Basic Python (variables, functions, loops)
- Basic SQL (SELECT, INSERT - we'll teach the rest!)
- Terminal/Command line basics

### System Requirements
- Ubuntu 20.04+ or Ubuntu-based Linux
- Python 3.8+
- ~1GB free disk space
- Internet connection

---

## 🛠️ Installation Guide

### Step 1: Update System

```bash
sudo apt update
sudo apt upgrade -y
```

---

### Step 2: Install PostgreSQL

PostgreSQL is the world's most advanced open-source relational database!

```bash
# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Check status
sudo systemctl status postgresql
```

You should see: `Active: active (running)`

---

### Step 3: Set Up PostgreSQL User

```bash
# Switch to postgres user
sudo -i -u postgres

# Create a new database user
createuser --interactive --pwprompt

# You'll be asked:
# Enter name of role to add: library_admin
# Enter password for new role: [choose a password]
# Shall the new role be a superuser? (y/n) y

# Create database
createdb -O library_admin library_db

# Exit postgres user
exit
```

**Save your password!** You'll need it later.

---

### Step 4: Test PostgreSQL Connection

```bash
# Connect to the database
psql -U library_admin -d library_db

# You should see:
# library_db=#

# Test a command
SELECT version();

# Exit
\q
```

---

### Step 5: Install Python Packages

```bash
# Create project folder
mkdir -p ~/library-management
cd ~/library-management

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install --upgrade pip
pip install psycopg2-binary sqlalchemy python-dotenv tabulate colorama
```

**Package explanations:**
- `psycopg2-binary` - PostgreSQL adapter for Python
- `sqlalchemy` - Python ORM (Object-Relational Mapping)
- `python-dotenv` - Load environment variables
- `tabulate` - Pretty print tables
- `colorama` - Colored terminal output

---

### Step 6: Create Environment File

```bash
nano .env
```

Add this (replace with YOUR password):

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=library_db
DB_USER=library_admin
DB_PASSWORD=your_password_here
```

Save: `Ctrl+O`, `Enter`, `Ctrl+X`

---

## 📁 Project Files

### File 1: `database.py` - Database Connection & Models

```bash
nano database.py
```

```python
"""
Database connection and table definitions using SQLAlchemy ORM.

This file defines the structure of our database tables and their relationships.
"""

from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, ForeignKey, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection string
DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

# Create database engine
engine = create_engine(DATABASE_URL, echo=False)  # Set echo=True to see SQL queries

# Create session factory
SessionLocal = sessionmaker(bind=engine)

# Base class for all models
Base = declarative_base()


# ==================== TABLE DEFINITIONS ====================

class Author(Base):
    """
    Authors table - stores information about book authors.
    
    Relationships:
    - Many-to-Many with Books (through book_authors)
    """
    __tablename__ = 'authors'
    
    author_id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    biography = Column(Text)
    birth_year = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationship to books (many-to-many)
    books = relationship('Book', secondary='book_authors', back_populates='authors')
    
    def __repr__(self):
        return f"<Author(id={self.author_id}, name={self.first_name} {self.last_name})>"


class Book(Base):
    """
    Books table - stores information about books in the library.
    
    Relationships:
    - Many-to-Many with Authors (through book_authors)
    - One-to-Many with Transactions
    """
    __tablename__ = 'books'
    
    book_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    isbn = Column(String(13), unique=True, nullable=False)
    publisher = Column(String(200))
    publish_year = Column(Integer)
    pages = Column(Integer)
    category = Column(String(100))
    description = Column(Text)
    total_copies = Column(Integer, default=1)
    available_copies = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    authors = relationship('Author', secondary='book_authors', back_populates='books')
    transactions = relationship('Transaction', back_populates='book')
    
    def __repr__(self):
        return f"<Book(id={self.book_id}, title={self.title})>"


class BookAuthor(Base):
    """
    Junction table for Many-to-Many relationship between Books and Authors.
    
    This table allows a book to have multiple authors and an author to write multiple books.
    """
    __tablename__ = 'book_authors'
    
    book_id = Column(Integer, ForeignKey('books.book_id', ondelete='CASCADE'), primary_key=True)
    author_id = Column(Integer, ForeignKey('authors.author_id', ondelete='CASCADE'), primary_key=True)


class Member(Base):
    """
    Members table - stores information about library members.
    
    Relationships:
    - One-to-Many with Transactions
    """
    __tablename__ = 'members'
    
    member_id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    phone = Column(String(15))
    address = Column(Text)
    join_date = Column(Date, default=datetime.now().date)
    membership_type = Column(String(50), default='Regular')  # Regular, Premium, Student
    status = Column(String(20), default='Active')  # Active, Suspended, Expired
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationship to transactions
    transactions = relationship('Transaction', back_populates='member')
    
    def __repr__(self):
        return f"<Member(id={self.member_id}, name={self.first_name} {self.last_name})>"


class Transaction(Base):
    """
    Transactions table - records book borrowing and returning.
    
    Relationships:
    - Many-to-One with Books
    - Many-to-One with Members
    """
    __tablename__ = 'transactions'
    
    transaction_id = Column(Integer, primary_key=True, autoincrement=True)
    book_id = Column(Integer, ForeignKey('books.book_id', ondelete='CASCADE'), nullable=False)
    member_id = Column(Integer, ForeignKey('members.member_id', ondelete='CASCADE'), nullable=False)
    borrow_date = Column(Date, default=datetime.now().date)
    due_date = Column(Date, nullable=False)
    return_date = Column(Date)
    status = Column(String(20), default='Borrowed')  # Borrowed, Returned, Overdue
    fine_amount = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    book = relationship('Book', back_populates='transactions')
    member = relationship('Member', back_populates='transactions')
    
    def __repr__(self):
        return f"<Transaction(id={self.transaction_id}, book_id={self.book_id}, member_id={self.member_id})>"


# ==================== DATABASE INITIALIZATION ====================

def init_database():
    """
    Create all tables in the database.
    This only needs to be run once when setting up the database.
    """
    print("🔧 Creating database tables...")
    Base.metadata.create_all(engine)
    print("✅ All tables created successfully!")


def get_session():
    """
    Get a new database session for executing queries.
    Always close the session when done!
    """
    return SessionLocal()


def test_connection():
    """Test if database connection works"""
    try:
        session = get_session()
        session.execute("SELECT 1")
        session.close()
        print("✅ Database connection successful!")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test connection
    test_connection()
    
    # Create tables
    init_database()
```

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

**What this file does:**
- Defines the structure of all 5 tables
- Sets up relationships between tables (Foreign Keys)
- Provides database connection functions
- Uses SQLAlchemy ORM (Object-Relational Mapping)

**Key Concepts:**
- `Primary Key (PK)` - Unique identifier for each row
- `Foreign Key (FK)` - Reference to another table's Primary Key
- `relationship()` - Defines how tables are connected
- `back_populates` - Creates bidirectional relationships

---

### File 2: `seed_data.py` - Sample Data

```bash
nano seed_data.py
```

```python
"""
Seed the database with sample data for testing.

This creates:
- 5 authors
- 10 books
- 5 members
- Some sample transactions
"""

from database import get_session, Author, Book, Member, Transaction
from datetime import datetime, timedelta
import random

def clear_database():
    """Clear all existing data"""
    session = get_session()
    try:
        session.query(Transaction).delete()
        session.query(Book).delete()
        session.query(Author).delete()
        session.query(Member).delete()
        session.commit()
        print("✅ Database cleared!")
    except Exception as e:
        session.rollback()
        print(f"❌ Error clearing database: {e}")
    finally:
        session.close()


def seed_authors():
    """Add sample authors"""
    session = get_session()
    
    authors_data = [
        {"first_name": "Robert", "last_name": "Martin", "biography": "Software engineer and author", "birth_year": 1952},
        {"first_name": "Martin", "last_name": "Fowler", "biography": "Chief Scientist at ThoughtWorks", "birth_year": 1963},
        {"first_name": "Eric", "last_name": "Evans", "biography": "Domain-Driven Design pioneer", "birth_year": 1960},
        {"first_name": "Kent", "last_name": "Beck", "biography": "Creator of Extreme Programming", "birth_year": 1961},
        {"first_name": "Andrew", "last_name": "Hunt", "biography": "Co-author of The Pragmatic Programmer", "birth_year": 1964},
    ]
    
    authors = []
    for data in authors_data:
        author = Author(**data)
        session.add(author)
        authors.append(author)
    
    session.commit()
    print(f"✅ Added {len(authors)} authors")
    
    author_ids = [a.author_id for a in authors]
    session.close()
    return author_ids


def seed_books(author_ids):
    """Add sample books"""
    session = get_session()
    
    books_data = [
        {"title": "Clean Code", "isbn": "9780132350884", "publisher": "Prentice Hall", "publish_year": 2008, "pages": 464, "category": "Programming", "description": "A Handbook of Agile Software Craftsmanship", "total_copies": 3, "available_copies": 3},
        {"title": "The Clean Coder", "isbn": "9780137081073", "publisher": "Prentice Hall", "publish_year": 2011, "pages": 256, "category": "Programming", "description": "A Code of Conduct for Professional Programmers", "total_copies": 2, "available_copies": 2},
        {"title": "Refactoring", "isbn": "9780201485677", "publisher": "Addison-Wesley", "publish_year": 1999, "pages": 464, "category": "Programming", "description": "Improving the Design of Existing Code", "total_copies": 2, "available_copies": 2},
        {"title": "Domain-Driven Design", "isbn": "9780321125217", "publisher": "Addison-Wesley", "publish_year": 2003, "pages": 560, "category": "Software Architecture", "description": "Tackling Complexity in the Heart of Software", "total_copies": 2, "available_copies": 2},
        {"title": "Test Driven Development", "isbn": "9780321146533", "publisher": "Addison-Wesley", "publish_year": 2002, "pages": 240, "category": "Programming", "description": "By Example", "total_copies": 2, "available_copies": 2},
        {"title": "The Pragmatic Programmer", "isbn": "9780135957059", "publisher": "Addison-Wesley", "publish_year": 2019, "pages": 352, "category": "Programming", "description": "Your Journey To Mastery", "total_copies": 4, "available_copies": 4},
        {"title": "Design Patterns", "isbn": "9780201633610", "publisher": "Addison-Wesley", "publish_year": 1994, "pages": 416, "category": "Software Architecture", "description": "Elements of Reusable Object-Oriented Software", "total_copies": 2, "available_copies": 2},
        {"title": "Clean Architecture", "isbn": "9780134494166", "publisher": "Prentice Hall", "publish_year": 2017, "pages": 432, "category": "Software Architecture", "description": "A Craftsman's Guide to Software Structure", "total_copies": 3, "available_copies": 3},
        {"title": "Continuous Delivery", "isbn": "9780321601919", "publisher": "Addison-Wesley", "publish_year": 2010, "pages": 512, "category": "DevOps", "description": "Reliable Software Releases", "total_copies": 2, "available_copies": 2},
        {"title": "The DevOps Handbook", "isbn": "9781942788003", "publisher": "IT Revolution Press", "publish_year": 2016, "pages": 480, "category": "DevOps", "description": "How to Create World-Class Agility", "total_copies": 2, "available_copies": 2},
    ]
    
    books = []
    for data in books_data:
        book = Book(**data)
        
        # Assign random authors to each book
        num_authors = random.randint(1, 2)
        book_authors = random.sample(author_ids, num_authors)
        
        for author_id in book_authors:
            author = session.query(Author).filter_by(author_id=author_id).first()
            book.authors.append(author)
        
        session.add(book)
        books.append(book)
    
    session.commit()
    print(f"✅ Added {len(books)} books")
    
    book_ids = [b.book_id for b in books]
    session.close()
    return book_ids


def seed_members():
    """Add sample members"""
    session = get_session()
    
    members_data = [
        {"first_name": "John", "last_name": "Doe", "email": "john.doe@email.com", "phone": "555-0001", "address": "123 Main St", "membership_type": "Regular"},
        {"first_name": "Jane", "last_name": "Smith", "email": "jane.smith@email.com", "phone": "555-0002", "address": "456 Oak Ave", "membership_type": "Premium"},
        {"first_name": "Bob", "last_name": "Johnson", "email": "bob.j@email.com", "phone": "555-0003", "address": "789 Pine Rd", "membership_type": "Student"},
        {"first_name": "Alice", "last_name": "Williams", "email": "alice.w@email.com", "phone": "555-0004", "address": "321 Elm St", "membership_type": "Regular"},
        {"first_name": "Charlie", "last_name": "Brown", "email": "charlie.b@email.com", "phone": "555-0005", "address": "654 Maple Dr", "membership_type": "Premium"},
    ]
    
    members = []
    for data in members_data:
        member = Member(**data)
        session.add(member)
        members.append(member)
    
    session.commit()
    print(f"✅ Added {len(members)} members")
    
    member_ids = [m.member_id for m in members]
    session.close()
    return member_ids


def seed_transactions(book_ids, member_ids):
    """Add sample transactions"""
    session = get_session()
    
    transactions = []
    
    # Create 3 active borrows
    for i in range(3):
        book_id = random.choice(book_ids)
        member_id = random.choice(member_ids)
        borrow_date = datetime.now().date() - timedelta(days=random.randint(1, 10))
        due_date = borrow_date + timedelta(days=14)
        
        transaction = Transaction(
            book_id=book_id,
            member_id=member_id,
            borrow_date=borrow_date,
            due_date=due_date,
            status='Borrowed'
        )
        session.add(transaction)
        transactions.append(transaction)
        
        # Update available copies
        book = session.query(Book).filter_by(book_id=book_id).first()
        book.available_copies -= 1
    
    # Create 2 returned transactions
    for i in range(2):
        book_id = random.choice(book_ids)
        member_id = random.choice(member_ids)
        borrow_date = datetime.now().date() - timedelta(days=random.randint(20, 30))
        due_date = borrow_date + timedelta(days=14)
        return_date = due_date - timedelta(days=random.randint(1, 5))
        
        transaction = Transaction(
            book_id=book_id,
            member_id=member_id,
            borrow_date=borrow_date,
            due_date=due_date,
            return_date=return_date,
            status='Returned'
        )
        session.add(transaction)
        transactions.append(transaction)
    
    session.commit()
    print(f"✅ Added {len(transactions)} transactions")
    session.close()


def seed_all():
    """Seed all data"""
    print("\n" + "="*60)
    print("🌱 Seeding Database with Sample Data")
    print("="*60 + "\n")
    
    # Clear existing data
    clear_database()
    
    # Seed new data
    author_ids = seed_authors()
    book_ids = seed_books(author_ids)
    member_ids = seed_members()
    seed_transactions(book_ids, member_ids)
    
    print("\n" + "="*60)
    print("✅ Database seeding complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    seed_all()
```

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

---

### File 3: `library_operations.py` - Core Operations

```bash
nano library_operations.py
```

```python
"""
Core library operations - add books, borrow, return, search, etc.

This file contains all the business logic for the library system.
"""

from database import get_session, Author, Book, Member, Transaction, BookAuthor
from datetime import datetime, timedelta
from sqlalchemy import func, and_, or_
from sqlalchemy.orm import joinedload

class LibraryOperations:
    """Main class for library operations"""
    
    def __init__(self):
        self.session = get_session()
    
    def close(self):
        """Close database session"""
        self.session.close()
    
    # ==================== BOOK OPERATIONS ====================
    
    def add_book(self, title, isbn, publisher, publish_year, pages, category, description, total_copies, author_ids):
        """
        Add a new book to the library.
        
        Args:
            title (str): Book title
            isbn (str): ISBN number
            publisher (str): Publisher name
            publish_year (int): Year of publication
            pages (int): Number of pages
            category (str): Book category
            description (str): Book description
            total_copies (int): Number of copies
            author_ids (list): List of author IDs
        
        Returns:
            Book object or None if error
        """
        try:
            # Check if ISBN already exists
            existing = self.session.query(Book).filter_by(isbn=isbn).first()
            if existing:
                print(f"❌ Book with ISBN {isbn} already exists!")
                return None
            
            # Create new book
            book = Book(
                title=title,
                isbn=isbn,
                publisher=publisher,
                publish_year=publish_year,
                pages=pages,
                category=category,
                description=description,
                total_copies=total_copies,
                available_copies=total_copies
            )
            
            # Add authors
            for author_id in author_ids:
                author = self.session.query(Author).filter_by(author_id=author_id).first()
                if author:
                    book.authors.append(author)
            
            self.session.add(book)
            self.session.commit()
            
            print(f"✅ Book '{title}' added successfully!")
            return book
            
        except Exception as e:
            self.session.rollback()
            print(f"❌ Error adding book: {e}")
            return None
    
    def search_books(self, query=None, category=None, author_name=None):
        """
        Search for books by title, category, or author.
        
        Args:
            query (str): Search query for title
            category (str): Filter by category
            author_name (str): Filter by author name
        
        Returns:
            List of Book objects
        """
        try:
            # Start with base query
            books_query = self.session.query(Book).options(joinedload(Book.authors))
            
            # Apply filters
            if query:
                books_query = books_query.filter(Book.title.ilike(f'%{query}%'))
            
            if category:
                books_query = books_query.filter(Book.category == category)
            
            if author_name:
                books_query = books_query.join(Book.authors).filter(
                    or_(
                        Author.first_name.ilike(f'%{author_name}%'),
                        Author.last_name.ilike(f'%{author_name}%')
                    )
                )
            
            books = books_query.all()
            return books
            
        except Exception as e:
            print(f"❌ Error searching books: {e}")
            return []
    
    def get_book_by_id(self, book_id):
        """Get a book by ID"""
        return self.session.query(Book).options(joinedload(Book.authors)).filter_by(book_id=book_id).first()
    
    # ==================== MEMBER OPERATIONS ====================
    
    def register_member(self, first_name, last_name, email, phone, address, membership_type='Regular'):
        """
        Register a new library member.
        
        Args:
            first_name (str): Member's first name
            last_name (str): Member's last name
            email (str): Member's email
            phone (str): Member's phone
            address (str): Member's address
            membership_type (str): Type of membership (Regular, Premium, Student)
        
        Returns:
            Member object or None if error
        """
        try:
            # Check if email already exists
            existing = self.session.query(Member).filter_by(email=email).first()
            if existing:
                print(f"❌ Member with email {email} already exists!")
                return None
            
            # Create new member
            member = Member(
                first_name=first_name,
                last_name=last_name,
                email=email,
                phone=phone,
                address=address,
                membership_type=membership_type,
                status='Active'
            )
            
            self.session.add(member)
            self.session.commit()
            
            print(f"✅ Member '{first_name} {last_name}' registered successfully!")
            return member
            
        except Exception as e:
            self.session.rollback()
            print(f"❌ Error registering member: {e}")
            return None
    
    def get_member_by_email(self, email):
        """Get a member by email"""
        return self.session.query(Member).filter_by(email=email).first()
    
    def get_member_by_id(self, member_id):
        """Get a member by ID"""
        return self.session.query(Member).filter_by(member_id=member_id).first()
    
    # ==================== TRANSACTION OPERATIONS ====================
    
    def borrow_book(self, book_id, member_id, days=14):
        """
        Borrow a book.
        
        Args:
            book_id (int): ID of the book to borrow
            member_id (int): ID of the member borrowing
            days (int): Number of days for the loan (default: 14)
        
        Returns:
            Transaction object or None if error
        """
        try:
            # Check if book exists and is available
            book = self.session.query(Book).filter_by(book_id=book_id).first()
            if not book:
                print(f"❌ Book with ID {book_id} not found!")
                return None
            
            if book.available_copies <= 0:
                print(f"❌ No copies of '{book.title}' are currently available!")
                return None
            
            # Check if member exists and is active
            member = self.session.query(Member).filter_by(member_id=member_id).first()
            if not member:
                print(f"❌ Member with ID {member_id} not found!")
                return None
            
            if member.status != 'Active':
                print(f"❌ Member account is not active!")
                return None
            
            # Check if member already has this book borrowed
            existing_borrow = self.session.query(Transaction).filter(
                and_(
                    Transaction.book_id == book_id,
                    Transaction.member_id == member_id,
                    Transaction.status == 'Borrowed'
                )
            ).first()
            
            if existing_borrow:
                print(f"❌ Member already has this book borrowed!")
                return None
            
            # Create transaction
            borrow_date = datetime.now().date()
            due_date = borrow_date + timedelta(days=days)
            
            transaction = Transaction(
                book_id=book_id,
                member_id=member_id,
                borrow_date=borrow_date,
                due_date=due_date,
                status='Borrowed'
            )
            
            # Update available copies
            book.available_copies -= 1
            
            self.session.add(transaction)
            self.session.commit()
            
            print(f"✅ Book '{book.title}' borrowed successfully!")
            print(f"📅 Due date: {due_date}")
            return transaction
            
        except Exception as e:
            self.session.rollback()
            print(f"❌ Error borrowing book: {e}")
            return None
    
    def return_book(self, transaction_id):
        """
        Return a borrowed book.
        
        Args:
            transaction_id (int): ID of the transaction
        
        Returns:
            Transaction object or None if error
        """
        try:
            # Find transaction
            transaction = self.session.query(Transaction).filter_by(transaction_id=transaction_id).first()
            if not transaction:
                print(f"❌ Transaction with ID {transaction_id} not found!")
                return None
            
            if transaction.status != 'Borrowed':
                print(f"❌ This book has already been returned!")
                return None
            
            # Calculate fine if overdue
            return_date = datetime.now().date()
            if return_date > transaction.due_date:
                days_overdue = (return_date - transaction.due_date).days
                fine_per_day = 0.50  # $0.50 per day
                transaction.fine_amount = days_overdue * fine_per_day
                transaction.status = 'Returned (Late)'
                print(f"⚠️  Book is {days_overdue} days overdue!")
                print(f"💰 Fine: ${transaction.fine_amount:.2f}")
            else:
                transaction.status = 'Returned'
            
            transaction.return_date = return_date
            
            # Update available copies
            book = self.session.query(Book).filter_by(book_id=transaction.book_id).first()
            book.available_copies += 1
            
            self.session.commit()
            
            print(f"✅ Book '{book.title}' returned successfully!")
            return transaction
            
        except Exception as e:
            self.session.rollback()
            print(f"❌ Error returning book: {e}")
            return None
    
    def get_member_borrowings(self, member_id):
        """Get all current borrowings for a member"""
        return self.session.query(Transaction).options(
            joinedload(Transaction.book),
            joinedload(Transaction.member)
        ).filter(
            and_(
                Transaction.member_id == member_id,
                Transaction.status == 'Borrowed'
            )
        ).all()
    
    def get_overdue_books(self):
        """
        Get all overdue books.
        
        Returns:
            List of Transaction objects
        """
        today = datetime.now().date()
        return self.session.query(Transaction).options(
            joinedload(Transaction.book),
            joinedload(Transaction.member)
        ).filter(
            and_(
                Transaction.status == 'Borrowed',
                Transaction.due_date < today
            )
        ).all()
    
    # ==================== REPORTS & ANALYTICS ====================
    
    def get_most_borrowed_books(self, limit=10):
        """
        Get the most popular books by borrow count.
        
        Args:
            limit (int): Number of books to return
        
        Returns:
            List of tuples (book, borrow_count)
        """
        results = self.session.query(
            Book,
            func.count(Transaction.transaction_id).label('borrow_count')
        ).join(Transaction).group_by(Book.book_id).order_by(
            func.count(Transaction.transaction_id).desc()
        ).limit(limit).all()
        
        return results
    
    def get_member_statistics(self, member_id):
        """
        Get borrowing statistics for a member.
        
        Args:
            member_id (int): Member ID
        
        Returns:
            Dictionary with statistics
        """
        member = self.session.query(Member).filter_by(member_id=member_id).first()
        if not member:
            return None
        
        total_borrowed = self.session.query(func.count(Transaction.transaction_id)).filter_by(
            member_id=member_id
        ).scalar()
        
        currently_borrowed = self.session.query(func.count(Transaction.transaction_id)).filter(
            and_(
                Transaction.member_id == member_id,
                Transaction.status == 'Borrowed'
            )
        ).scalar()
        
        total_fines = self.session.query(func.sum(Transaction.fine_amount)).filter_by(
            member_id=member_id
        ).scalar() or 0
        
        return {
            'member': member,
            'total_borrowed': total_borrowed,
            'currently_borrowed': currently_borrowed,
            'total_fines': total_fines
        }
    
    def get_category_distribution(self):
        """Get distribution of books by category"""
        return self.session.query(
            Book.category,
            func.count(Book.book_id).label('count')
        ).group_by(Book.category).order_by(
            func.count(Book.book_id).desc()
        ).all()
    
    # ==================== AUTHOR OPERATIONS ====================
    
    def add_author(self, first_name, last_name, biography=None, birth_year=None):
        """
        Add a new author.
        
        Args:
            first_name (str): Author's first name
            last_name (str): Author's last name
            biography (str): Author's biography
            birth_year (int): Author's birth year
        
        Returns:
            Author object or None if error
        """
        try:
            author = Author(
                first_name=first_name,
                last_name=last_name,
                biography=biography,
                birth_year=birth_year
            )
            
            self.session.add(author)
            self.session.commit()
            
            print(f"✅ Author '{first_name} {last_name}' added successfully!")
            return author
            
        except Exception as e:
            self.session.rollback()
            print(f"❌ Error adding author: {e}")
            return None
    
    def list_all_authors(self):
        """Get all authors"""
        return self.session.query(Author).order_by(Author.last_name).all()


if __name__ == "__main__":
    # Test operations
    ops = LibraryOperations()
    
    # Test search
    print("\n📚 Searching for 'Clean' books:")
    books = ops.search_books(query='Clean')
    for book in books:
        authors = ", ".join([f"{a.first_name} {a.last_name}" for a in book.authors])
        print(f"  - {book.title} by {authors}")
    
    ops.close()
```

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

---

### File 4: `main.py` - Command Line Interface

```bash
nano main.py
```

```python
"""
Main command-line interface for the Library Management System.

This provides an interactive menu for users to interact with the library.
"""

from library_operations import LibraryOperations
from tabulate import tabulate
from colorama import init, Fore, Style
import sys

# Initialize colorama for colored output
init(autoreset=True)

class LibraryMenu:
    """Interactive menu system for library operations"""
    
    def __init__(self):
        self.ops = LibraryOperations()
    
    def display_header(self):
        """Display application header"""
        print("\n" + "="*70)
        print(Fore.CYAN + Style.BRIGHT + "📚 CITY LIBRARY MANAGEMENT SYSTEM 📚".center(70))
        print("="*70 + "\n")
    
    def display_menu(self):
        """Display main menu"""
        print(Fore.YELLOW + "\n╔════════════════════ MAIN MENU ════════════════════╗")
        print(Fore.YELLOW + "║                                                   ║")
        print(Fore.GREEN + "║  1. 📖  Add New Book                              ║")
        print(Fore.GREEN + "║  2. ✍️   Add New Author                            ║")
        print(Fore.GREEN + "║  3. 👤  Register New Member                       ║")
        print(Fore.BLUE + "║  4. 🔍  Search Books                              ║")
        print(Fore.BLUE + "║  5. 📚  View All Books                            ║")
        print(Fore.BLUE + "║  6. 👥  View All Members                          ║")
        print(Fore.MAGENTA + "║  7. 📤  Borrow a Book                             ║")
        print(Fore.MAGENTA + "║  8. 📥  Return a Book                             ║")
        print(Fore.MAGENTA + "║  9. 🕐  View Member's Borrowings                  ║")
        print(Fore.RED + "║  10. ⚠️   View Overdue Books                       ║")
        print(Fore.CYAN + "║  11. 📊  View Reports                             ║")
        print(Fore.YELLOW + "║  12. 🚪  Exit                                     ║")
        print(Fore.YELLOW + "║                                                   ║")
        print(Fore.YELLOW + "╚═══════════════════════════════════════════════════╝\n")
    
    def add_book_menu(self):
        """Menu for adding a new book"""
        print(Fore.CYAN + "\n➕ ADD NEW BOOK\n")
        
        try:
            title = input("Enter book title: ").strip()
            isbn = input("Enter ISBN (13 digits): ").strip()
            publisher = input("Enter publisher: ").strip()
            publish_year = int(input("Enter publish year: "))
            pages = int(input("Enter number of pages: "))
            category = input("Enter category: ").strip()
            description = input("Enter description: ").strip()
            total_copies = int(input("Enter total copies: "))
            
            # Show available authors
            authors = self.ops.list_all_authors()
            print("\n📋 Available Authors:")
            author_table = [[a.author_id, f"{a.first_name} {a.last_name}"] for a in authors]
            print(tabulate(author_table, headers=["ID", "Name"], tablefmt="grid"))
            
            author_ids = input("\nEnter author IDs (comma-separated): ").strip()
            author_ids = [int(x.strip()) for x in author_ids.split(',')]
            
            self.ops.add_book(title, isbn, publisher, publish_year, pages, category, description, total_copies, author_ids)
            
        except ValueError as e:
            print(Fore.RED + f"❌ Invalid input: {e}")
        except Exception as e:
            print(Fore.RED + f"❌ Error: {e}")
    
    def add_author_menu(self):
        """Menu for adding a new author"""
        print(Fore.CYAN + "\n➕ ADD NEW AUTHOR\n")
        
        try:
            first_name = input("Enter first name: ").strip()
            last_name = input("Enter last name: ").strip()
            biography = input("Enter biography (optional): ").strip() or None
            birth_year_str = input("Enter birth year (optional): ").strip()
            birth_year = int(birth_year_str) if birth_year_str else None
            
            self.ops.add_author(first_name, last_name, biography, birth_year)
            
        except ValueError as e:
            print(Fore.RED + f"❌ Invalid input: {e}")
        except Exception as e:
            print(Fore.RED + f"❌ Error: {e}")
    
    def register_member_menu(self):
        """Menu for registering a new member"""
        print(Fore.CYAN + "\n➕ REGISTER NEW MEMBER\n")
        
        try:
            first_name = input("Enter first name: ").strip()
            last_name = input("Enter last name: ").strip()
            email = input("Enter email: ").strip()
            phone = input("Enter phone: ").strip()
            address = input("Enter address: ").strip()
            
            print("\nMembership Types:")
            print("1. Regular")
            print("2. Premium")
            print("3. Student")
            choice = input("Select membership type (1-3): ").strip()
            
            membership_types = {'1': 'Regular', '2': 'Premium', '3': 'Student'}
            membership_type = membership_types.get(choice, 'Regular')
            
            self.ops.register_member(first_name, last_name, email, phone, address, membership_type)
            
        except Exception as e:
            print(Fore.RED + f"❌ Error: {e}")
    
    def search_books_menu(self):
        """Menu for searching books"""
        print(Fore.CYAN + "\n🔍 SEARCH BOOKS\n")
        
        print("Search by:")
        print("1. Title")
        print("2. Category")
        print("3. Author")
        print("4. All")
        
        choice = input("\nSelect search type (1-4): ").strip()
        
        query = None
        category = None
        author_name = None
        
        if choice == '1':
            query = input("Enter title keyword: ").strip()
        elif choice == '2':
            category = input("Enter category: ").strip()
        elif choice == '3':
            author_name = input("Enter author name: ").strip()
        
        books = self.ops.search_books(query, category, author_name)
        
        if books:
            print(Fore.GREEN + f"\n✅ Found {len(books)} book(s):\n")
            self.display_books(books)
        else:
            print(Fore.YELLOW + "\n⚠️  No books found!")
    
    def display_books(self, books):
        """Display books in a table"""
        table_data = []
        for book in books:
            authors = ", ".join([f"{a.first_name} {a.last_name}" for a in book.authors])
            table_data.append([
                book.book_id,
                book.title[:40],
                authors[:30],
                book.category,
                book.publish_year,
                f"{book.available_copies}/{book.total_copies}"
            ])
        
        print(tabulate(table_data, 
                      headers=["ID", "Title", "Author(s)", "Category", "Year", "Available"],
                      tablefmt="grid"))
    
    def view_all_books_menu(self):
        """View all books in the library"""
        print(Fore.CYAN + "\n📚 ALL BOOKS\n")
        
        books = self.ops.search_books()
        if books:
            self.display_books(books)
        else:
            print(Fore.YELLOW + "⚠️  No books in library!")
    
    def view_all_members_menu(self):
        """View all members"""
        print(Fore.CYAN + "\n👥 ALL MEMBERS\n")
        
        session = self.ops.session
        members = session.query(self.ops.session.query(Member)).all()
        
        if members:
            table_data = [[
                m.member_id,
                f"{m.first_name} {m.last_name}",
                m.email,
                m.membership_type,
                m.status,
                m.join_date
            ] for m in members]
            
            print(tabulate(table_data,
                          headers=["ID", "Name", "Email", "Type", "Status", "Join Date"],
                          tablefmt="grid"))
        else:
            print(Fore.YELLOW + "⚠️  No members registered!")
    
    def borrow_book_menu(self):
        """Menu for borrowing a book"""
        print(Fore.CYAN + "\n📤 BORROW A BOOK\n")
        
        try:
            book_id = int(input("Enter book ID: "))
            member_id = int(input("Enter member ID: "))
            
            days_input = input("Enter loan period in days (default: 14): ").strip()
            days = int(days_input) if days_input else 14
            
            self.ops.borrow_book(book_id, member_id, days)
            
        except ValueError:
            print(Fore.RED + "❌ Invalid input! Please enter numbers only.")
        except Exception as e:
            print(Fore.RED + f"❌ Error: {e}")
    
    def return_book_menu(self):
        """Menu for returning a book"""
        print(Fore.CYAN + "\n📥 RETURN A BOOK\n")
        
        try:
            transaction_id = int(input("Enter transaction ID: "))
            self.ops.return_book(transaction_id)
            
        except ValueError:
            print(Fore.RED + "❌ Invalid input! Please enter a number.")
        except Exception as e:
            print(Fore.RED + f"❌ Error: {e}")
    
    def view_member_borrowings_menu(self):
        """View a member's current borrowings"""
        print(Fore.CYAN + "\n🕐 MEMBER'S BORROWINGS\n")
        
        try:
            member_id = int(input("Enter member ID: "))
            borrowings = self.ops.get_member_borrowings(member_id)
            
            if borrowings:
                table_data = [[
                    t.transaction_id,
                    t.book.title[:40],
                    t.borrow_date,
                    t.due_date,
                    t.status
                ] for t in borrowings]
                
                print(tabulate(table_data,
                              headers=["Transaction ID", "Book", "Borrowed", "Due Date", "Status"],
                              tablefmt="grid"))
            else:
                print(Fore.YELLOW + "⚠️  No current borrowings for this member!")
                
        except ValueError:
            print(Fore.RED + "❌ Invalid input! Please enter a number.")
        except Exception as e:
            print(Fore.RED + f"❌ Error: {e}")
    
    def view_overdue_books_menu(self):
        """View all overdue books"""
        print(Fore.RED + "\n⚠️  OVERDUE BOOKS\n")
        
        overdue = self.ops.get_overdue_books()
        
        if overdue:
            table_data = [[
                t.transaction_id,
                t.book.title[:30],
                f"{t.member.first_name} {t.member.last_name}",
                t.due_date,
                (t.due_date - datetime.now().date()).days
            ] for t in overdue]
            
            print(tabulate(table_data,
                          headers=["Trans ID", "Book", "Member", "Due Date", "Days Overdue"],
                          tablefmt="grid"))
        else:
            print(Fore.GREEN + "✅ No overdue books!")
    
    def view_reports_menu(self):
        """View various reports"""
        print(Fore.CYAN + "\n📊 REPORTS\n")
        
        print("1. Most Borrowed Books")
        print("2. Member Statistics")
        print("3. Category Distribution")
        
        choice = input("\nSelect report (1-3): ").strip()
        
        if choice == '1':
            results = self.ops.get_most_borrowed_books(10)
            if results:
                table_data = [[
                    book.title[:40],
                    ", ".join([f"{a.first_name} {a.last_name}" for a in book.authors])[:30],
                    count
                ] for book, count in results]
                
                print(Fore.GREEN + "\n📚 TOP 10 MOST BORROWED BOOKS:\n")
                print(tabulate(table_data,
                              headers=["Title", "Author(s)", "Times Borrowed"],
                              tablefmt="grid"))
        
        elif choice == '2':
            try:
                member_id = int(input("Enter member ID: "))
                stats = self.ops.get_member_statistics(member_id)
                
                if stats:
                    print(Fore.GREEN + f"\n📊 Statistics for {stats['member'].first_name} {stats['member'].last_name}:\n")
                    print(f"Total books borrowed: {stats['total_borrowed']}")
                    print(f"Currently borrowed: {stats['currently_borrowed']}")
                    print(f"Total fines: ${stats['total_fines']:.2f}")
                else:
                    print(Fore.YELLOW + "⚠️  Member not found!")
            except ValueError:
                print(Fore.RED + "❌ Invalid input!")
        
        elif choice == '3':
            results = self.ops.get_category_distribution()
            if results:
                table_data = [[category, count] for category, count in results]
                
                print(Fore.GREEN + "\n📊 BOOKS BY CATEGORY:\n")
                print(tabulate(table_data,
                              headers=["Category", "Number of Books"],
                              tablefmt="grid"))
    
    def run(self):
        """Main application loop"""
        try:
            while True:
                self.display_header()
                self.display_menu()
                
                choice = input(Fore.YELLOW + "Enter your choice (1-12): ").strip()
                
                if choice == '1':
                    self.add_book_menu()
                elif choice == '2':
                    self.add_author_menu()
                elif choice == '3':
                    self.register_member_menu()
                elif choice == '4':
                    self.search_books_menu()
                elif choice == '5':
                    self.view_all_books_menu()
                elif choice == '6':
                    self.view_all_members_menu()
                elif choice == '7':
                    self.borrow_book_menu()
                elif choice == '8':
                    self.return_book_menu()
                elif choice == '9':
                    self.view_member_borrowings_menu()
                elif choice == '10':
                    self.view_overdue_books_menu()
                elif choice == '11':
                    self.view_reports_menu()
                elif choice == '12':
                    print(Fore.GREEN + "\n👋 Thank you for using City Library Management System!")
                    break
                else:
                    print(Fore.RED + "\n❌ Invalid choice! Please try again.")
                
                input(Fore.YELLOW + "\nPress Enter to continue...")
        
        except KeyboardInterrupt:
            print(Fore.YELLOW + "\n\n👋 Goodbye!")
        finally:
            self.ops.close()


if __name__ == "__main__":
    from datetime import datetime
    from database import Member
    
    menu = LibraryMenu()
    menu.run()
```

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

---

### File 5: `requirements.txt` - Dependencies

```bash
nano requirements.txt
```

```txt
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0
python-dotenv>=1.0.0
tabulate>=0.9.0
colorama>=0.4.6
```

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

---

### File 6: `.gitignore`

```bash
nano .gitignore
```

```
# Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd

# Environment
.env

# Database backups
*.sql
*.dump

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
```

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

---

## 🚀 Running the Project

### Step 1: Initialize Database

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Create tables
python3 database.py
```

**Expected output:**
```
✅ Database connection successful!
🔧 Creating database tables...
✅ All tables created successfully!
```

---

### Step 2: Seed Sample Data

```bash
python3 seed_data.py
```

**Expected output:**
```
============================================================
🌱 Seeding Database with Sample Data
============================================================

✅ Database cleared!
✅ Added 5 authors
✅ Added 10 books
✅ Added 5 members
✅ Added 5 transactions

============================================================
✅ Database seeding complete!
============================================================
```

---

### Step 3: Run the Application

```bash
python3 main.py
```

You'll see the interactive menu! 🎉

---

## 📝 SQL Queries Explained

Let's understand the SQL behind key operations:

### 1. **Simple SELECT** - Get all books
```sql
SELECT * FROM books;
```

### 2. **JOIN** - Get books with their authors
```sql
SELECT 
    b.title,
    a.first_name,
    a.last_name
FROM books b
INNER JOIN book_authors ba ON b.book_id = ba.book_id
INNER JOIN authors a ON ba.author_id = a.author_id;
```

### 3. **WHERE** - Find available books
```sql
SELECT * FROM books WHERE available_copies > 0;
```

### 4. **GROUP BY** - Count books by category
```sql
SELECT 
    category,
    COUNT(*) as book_count
FROM books
GROUP BY category
ORDER BY book_count DESC;
```

### 5. **Complex JOIN** - Find overdue books
```sql
SELECT 
    t.transaction_id,
    b.title,
    m.first_name || ' ' || m.last_name as member_name,
    t.due_date,
    CURRENT_DATE - t.due_date as days_overdue
FROM transactions t
INNER JOIN books b ON t.book_id = b.book_id
INNER JOIN members m ON t.member_id = m.member_id
WHERE t.status = 'Borrowed'
    AND t.due_date < CURRENT_DATE
ORDER BY days_overdue DESC;
```

### 6. **Subquery** - Find members who borrowed more than 3 books
```sql
SELECT 
    m.first_name,
    m.last_name,
    (SELECT COUNT(*) 
     FROM transactions t 
     WHERE t.member_id = m.member_id) as total_borrows
FROM members m
WHERE (SELECT COUNT(*) 
       FROM transactions t 
       WHERE t.member_id = m.member_id) > 3;
```

### 7. **Aggregate Functions** - Most popular book
```sql
SELECT 
    b.title,
    COUNT(t.transaction_id) as borrow_count
FROM books b
INNER JOIN transactions t ON b.book_id = t.book_id
GROUP BY b.book_id, b.title
ORDER BY borrow_count DESC
LIMIT 1;
```

### 8. **INSERT with Foreign Keys**
```sql
-- First, add a book
INSERT INTO books (title, isbn, publisher, publish_year, total_copies, available_copies)
VALUES ('New Book', '1234567890123', 'Publisher Inc', 2024, 5, 5)
RETURNING book_id;

-- Then link it to authors
INSERT INTO book_authors (book_id, author_id)
VALUES (1, 1), (1, 2);
```

### 9. **UPDATE** - Return a book
```sql
-- Update transaction
UPDATE transactions
SET 
    return_date = CURRENT_DATE,
    status = 'Returned'
WHERE transaction_id = 1;

-- Update book availability
UPDATE books
SET available_copies = available_copies + 1
WHERE book_id = (SELECT book_id FROM transactions WHERE transaction_id = 1);
```

### 10. **Transaction** - Borrow a book (atomic operation)
```sql
BEGIN;

-- Check availability
SELECT available_copies FROM books WHERE book_id = 1 FOR UPDATE;

-- If available, create transaction
INSERT INTO transactions (book_id, member_id, borrow_date, due_date, status)
VALUES (1, 1, CURRENT_DATE, CURRENT_DATE + INTERVAL '14 days', 'Borrowed');

-- Update availability
UPDATE books SET available_copies = available_copies - 1 WHERE book_id = 1;

COMMIT;
```

---

## 🎓 Advanced Features

### Feature 1: Create a View for Popular Books

```bash
# Connect to PostgreSQL
psql -U library_admin -d library_db
```

```sql
-- Create a view
CREATE VIEW popular_books AS
SELECT 
    b.book_id,
    b.title,
    STRING_AGG(a.first_name || ' ' || a.last_name, ', ') as authors,
    COUNT(t.transaction_id) as times_borrowed
FROM books b
LEFT JOIN book_authors ba ON b.book_id = ba.book_id
LEFT JOIN authors a ON ba.author_id = a.author_id
LEFT JOIN transactions t ON b.book_id = t.book_id
GROUP BY b.book_id, b.title
ORDER BY times_borrowed DESC;

-- Use the view
SELECT * FROM popular_books LIMIT 5;
```

---

### Feature 2: Create Indexes for Fast Searches

```sql
-- Index on book title (for faster searches)
CREATE INDEX idx_books_title ON books(title);

-- Index on ISBN (unique identifier)
CREATE INDEX idx_books_isbn ON books(isbn);

-- Index on member email
CREATE INDEX idx_members_email ON members(email);

-- Composite index for transactions
CREATE INDEX idx_transactions_status_due ON transactions(status, due_date);

-- Show all indexes
\di
```

**Why indexes?**
- Without index: Database scans EVERY row
- With index: Database jumps directly to matching rows
- Searches become 100x faster!

---

### Feature 3: Add Triggers for Automatic Actions

```sql
-- Create a function to update book status automatically
CREATE OR REPLACE FUNCTION update_transaction_status()
RETURNS TRIGGER AS $
BEGIN
    IF NEW.status = 'Borrowed' AND NEW.due_date < CURRENT_DATE THEN
        NEW.status = 'Overdue';
    END IF;
    RETURN NEW;
END;
$ LANGUAGE plpgsql;

-- Create trigger
CREATE TRIGGER check_overdue
    BEFORE UPDATE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_transaction_status();
```

---

### Feature 4: Backup and Restore

**Backup database:**
```bash
# Full backup
pg_dump -U library_admin library_db > library_backup_$(date +%Y%m%d).sql

# Backup only data
pg_dump -U library_admin --data-only library_db > library_data_$(date +%Y%m%d).sql

# Backup only schema
pg_dump -U library_admin --schema-only library_db > library_schema.sql
```

**Restore database:**
```bash
# Drop existing database (careful!)
dropdb -U library_admin library_db

# Create new database
createdb -U library_admin library_db

# Restore from backup
psql -U library_admin library_db < library_backup_20240101.sql
```

---

### Feature 5: Query Optimization

**Before optimization:**
```sql
-- Slow query
SELECT * FROM books
WHERE book_id IN (
    SELECT book_id FROM transactions WHERE status = 'Borrowed'
);
```

**After optimization:**
```sql
-- Fast query using JOIN
SELECT DISTINCT b.*
FROM books b
INNER JOIN transactions t ON b.book_id = t.book_id
WHERE t.status = 'Borrowed';
```

**Explain query performance:**
```sql
EXPLAIN ANALYZE
SELECT * FROM books WHERE title ILIKE '%clean%';
```

---

## 🐛 Troubleshooting

### Problem: "psycopg2.OperationalError: could not connect"

**Solution:**
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# If not running, start it
sudo systemctl start postgresql

# Check if you can connect manually
psql -U library_admin -d library_db

# If password fails, reset it:
sudo -i -u postgres
psql
ALTER USER library_admin WITH PASSWORD 'new_password';
\q
exit

# Update .env file with new password
```

---

### Problem: "relation does not exist"

**Solution:**
```bash
# Tables not created yet
python3 database.py

# Or create manually:
psql -U library_admin -d library_db
# Then paste SQL from database.py CREATE TABLE statements
```

---

### Problem: "duplicate key value violates unique constraint"

**Solution:**
This means you're trying to insert duplicate data (e.g., same ISBN twice).

```python
# Check if exists first
existing = session.query(Book).filter_by(isbn=isbn).first()
if existing:
    print("Book already exists!")
else:
    # Add new book
```

---

### Problem: "foreign key constraint violation"

**Solution:**
You're trying to reference a non-existent ID.

```python
# Verify the referenced record exists
author = session.query(Author).filter_by(author_id=author_id).first()
if not author:
    print(f"Author with ID {author_id} does not exist!")
```

---

### Problem: "permission denied for database"

**Solution:**
```bash
# Grant permissions
sudo -i -u postgres
psql
GRANT ALL PRIVILEGES ON DATABASE library_db TO library_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO library_admin;
\q
exit
```

---

### Problem: PostgreSQL using too much memory

**Solution:**
```bash
# Edit PostgreSQL config
sudo nano /etc/postgresql/13/main/postgresql.conf

# Adjust these settings:
shared_buffers = 256MB          # 25% of RAM
work_mem = 4MB
maintenance_work_mem = 64MB
effective_cache_size = 1GB      # 50-75% of RAM

# Restart PostgreSQL
sudo systemctl restart postgresql
```

---

## 📊 Understanding Database Design Principles

### 1. **Normalization** (Avoiding Data Redundancy)

**❌ Bad Design (Unnormalized):**
```
Books Table:
| book_id | title      | author1_name | author1_bio | author2_name | author2_bio |
```
**Problems:**
- Limited to 2 authors
- Author info duplicated for each book
- Can't have authors without books

**✅ Good Design (Normalized):**
```
Books Table:        Authors Table:      BookAuthors Table:
| book_id |         | author_id |       | book_id | author_id |
| title   |         | name      |       
                    | bio       |
```

---

### 2. **ACID Properties** (Database Transactions)

- **Atomicity**: All-or-nothing (either all changes happen, or none do)
- **Consistency**: Database stays in valid state
- **Isolation**: Concurrent transactions don't interfere
- **Durability**: Committed data persists even if system crashes

**Example:**
```python
# Borrowing a book - all 3 operations must succeed
try:
    # 1. Create transaction
    transaction = Transaction(...)
    session.add(transaction)
    
    # 2. Decrease available copies
    book.available_copies -= 1
    
    # 3. Save everything
    session.commit()  # ✅ All changes saved
except:
    session.rollback()  # ❌ Nothing changes
```

---

### 3. **Cardinality** (Relationship Types)

**One-to-One (1:1):**
```
Member ←→ MemberProfile
One member has exactly one profile
```

**One-to-Many (1:N):**
```
Member ←→ Transactions
One member can have many transactions
```

**Many-to-Many (M:N):**
```
Books ←→ Authors (via BookAuthors junction table)
A book can have many authors
An author can write many books
```

---

### 4. **Constraints** (Data Validation)

```sql
-- Primary Key: Unique identifier
book_id INTEGER PRIMARY KEY

-- Foreign Key: Must reference existing record
member_id INTEGER REFERENCES members(member_id)

-- NOT NULL: Field must have a value
email VARCHAR(255) NOT NULL

-- UNIQUE: No duplicates allowed
isbn VARCHAR(13) UNIQUE

-- CHECK: Custom validation
available_copies INTEGER CHECK (available_copies >= 0)

-- DEFAULT: Automatic value
created_at TIMESTAMP DEFAULT NOW()
```

---

## 🎯 Practice Exercises

### Exercise 1: Add a Reviews Table
Add a feature for members to review books.

**Hints:**
```sql
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    book_id INTEGER REFERENCES books(book_id),
    member_id INTEGER REFERENCES members(member_id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

### Exercise 2: Late Fee Calculation
Implement automatic late fee calculation.

**Hints:**
- Fine = $0.50 per day overdue
- Premium members get 50% discount
- Update `return_book()` function

---

### Exercise 3: Reservation System
Allow members to reserve books that are currently borrowed.

**Hints:**
```sql
CREATE TABLE reservations (
    reservation_id SERIAL PRIMARY KEY,
    book_id INTEGER REFERENCES books(book_id),
    member_id INTEGER REFERENCES members(member_id),
    reservation_date DATE DEFAULT CURRENT_DATE,
    status VARCHAR(20) DEFAULT 'Pending'
);
```

---

### Exercise 4: Email Notifications
Send email reminders for due dates.

**Hints:**
- Use Python's `smtplib` library
- Create a scheduled task to check daily
- Send email 2 days before due date

---

## 📚 Learning Path

### Beginner Level ✅
- [x] Understand tables and columns
- [x] Basic SQL: SELECT, INSERT, UPDATE, DELETE
- [x] Primary and Foreign Keys
- [x] Simple JOINs

### Intermediate Level
- [ ] Complex JOINs (LEFT, RIGHT, FULL OUTER)
- [ ] Subqueries and CTEs
- [ ] Indexes and Query Optimization
- [ ] Transactions and ACID properties
- [ ] Views and Stored Procedures

### Advanced Level
- [ ] Database normalization (1NF, 2NF, 3NF, BCNF)
- [ ] Triggers and Functions
- [ ] Partitioning and Sharding
- [ ] Replication and High Availability
- [ ] Performance Tuning

---

## 🔗 Useful PostgreSQL Commands

```bash
# Connect to database
psql -U library_admin -d library_db

# List all databases
\l

# List all tables
\dt

# Describe table structure
\d books

# List all indexes
\di

# Show table data
SELECT * FROM books LIMIT 10;

# Export query results to CSV
\copy (SELECT * FROM books) TO '/tmp/books.csv' CSV HEADER;

# Execute SQL file
\i /path/to/file.sql

# Show query execution time
\timing on

# Quit
\q
```

---

## 📖 Additional Resources

### Official Documentation
- **PostgreSQL Docs**: https://www.postgresql.org/docs/
- **SQLAlchemy Docs**: https://docs.sqlalchemy.org/

### Learning Resources
- **SQL Tutorial**: https://www.w3schools.com/sql/
- **PostgreSQL Tutorial**: https://www.postgresqltutorial.com/
- **Database Design**: https://www.lucidchart.com/pages/database-diagram/database-design

### Practice
- **SQLZoo**: https://sqlzoo.net/ (Interactive SQL exercises)
- **LeetCode Database**: https://leetcode.com/problemset/database/
- **HackerRank SQL**: https://www.hackerrank.com/domains/sql

---

## 🎉 Congratulations!

You've built a complete library management system and learned:

✅ **Database Design** - How to structure data properly
✅ **SQL Fundamentals** - Writing queries to manipulate data
✅ **Relationships** - Connecting tables with Foreign Keys
✅ **Transactions** - Ensuring data consistency
✅ **PostgreSQL** - Industry-standard relational database
✅ **SQLAlchemy ORM** - Python database toolkit
✅ **Real-World Application** - Practical problem solving

---

## 🚀 Next Steps

1. **Add more features**: Reservations, Reviews, Email notifications
2. **Build a web interface**: Use Flask or Django
3. **Deploy online**: Use Heroku or DigitalOcean
4. **Learn advanced SQL**: Window functions, CTEs, Recursive queries
5. **Explore other databases**: MySQL, MongoDB, Redis

---

## 📝 Project Checklist

- [ ] PostgreSQL installed and running
- [ ] Database and user created
- [ ] Virtual environment set up
- [ ] All packages installed
- [ ] `.env` file configured
- [ ] All 6 Python files created
- [ ] Database tables created (`python3 database.py`)
- [ ] Sample data seeded (`python3 seed_data.py`)
- [ ] Application runs (`python3 main.py`)
- [ ] Can add books, members, authors
- [ ] Can borrow and return books
- [ ] Can search and generate reports
- [ ] Understand all SQL queries
- [ ] Completed at least one practice exercise

---

## 💡 Pro Tips

1. **Always use transactions** for operations that modify multiple tables
2. **Create indexes** on columns you frequently search
3. **Use EXPLAIN ANALYZE** to understand query performance
4. **Regular backups** - Database corruption happens!
5. **Validate input** at both application AND database level
6. **Use connection pooling** for better performance
7. **Document your schema** - Future you will thank you!
8. **Test with realistic data** - Seed lots of sample data
9. **Monitor slow queries** - PostgreSQL has built-in logging
10. **Keep learning** - Databases are deep and fascinating!

---

## 🤝 Contributing

Want to improve this project?
- Add new features (reviews, wishlists, etc.)
- Improve the UI (add colors, better formatting)
- Write tests (pytest)
- Add API endpoints (Flask/FastAPI)
- Create a web dashboard

---



---

**Happy coding! May your queries be fast and your data consistent! 🚀📚💾**

---

*Made with ❤️ for aspiring database developers*
*Last updated: 2025*
