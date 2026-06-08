import sqlite3

def main():
    conn = sqlite3.connect('facts.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, keywords TEXT, content TEXT)')
    c.execute('DELETE FROM facts') # Clear old data
    
    facts = [
        ("argyle founded", "The Argyle company was founded in 2026 by Sparsh."),
        ("secret vault code", "The secret vault code is 8842."),
        ("ceo division new", "The CEO of the new division is Sparsh."),
        ("capital france", "Paris is the capital of France.")
    ]
    
    c.executemany('INSERT INTO facts (keywords, content) VALUES (?, ?)', facts)
    conn.commit()
    conn.close()
    print("Created facts.db with 4 classified facts.")

if __name__ == '__main__':
    main()
