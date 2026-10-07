from src.data import db_models  # noqa: F401
from src.data.database import Base, engine


def initialize_database() -> None:
    """
    Create all application database tables.
    """
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    initialize_database()
    print("Database tables created successfully.")