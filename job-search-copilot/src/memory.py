# SQLite Schema:
#
# CREATE TABLE jobs (
#     id INTEGER PRIMARY KEY,
#     title TEXT,
#     company TEXT,
#     link TEXT,
#     description TEXT,
#     fetched_at TIMESTAMP
# );
#
# CREATE TABLE resumes (
#     id INTEGER PRIMARY KEY,
#     filename TEXT,
#     content TEXT,
#     uploaded_at TIMESTAMP
# );
#
# CREATE TABLE interviews (
#     id INTEGER PRIMARY KEY,
#     job_id INTEGER,
#     notes TEXT,
#     created_at TIMESTAMP
# );
