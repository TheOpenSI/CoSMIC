import os
import tempfile
from sqlalchemy import create_engine, Column, String, BigInteger
from sqlalchemy.orm import sessionmaker

# Point CoSMIC at a temp sqlite for its own DB
cosmic_fd, cosmic_path = tempfile.mkstemp(prefix="cosmic_", suffix=".db")
os.close(cosmic_fd)
os.environ["COSMIC_DB_URL"] = f"sqlite:///{cosmic_path}"

# Build a tiny OpenWebUI sqlite with a `user` table
ow_fd, ow_path = tempfile.mkstemp(prefix="ow_", suffix=".db")
os.close(ow_fd)
ow_url = f"sqlite:///{ow_path}"
os.environ["OPENWEBUI_DATABASE_URL"] = ow_url

ow_engine = create_engine(ow_url)

from sqlalchemy.orm import declarative_base
Base = declarative_base()

class OWUser(Base):
    __tablename__ = "user"
    id = Column(String, primary_key=True)
    name = Column(String)
    email = Column(String)
    role = Column(String)
    profile_image_url = Column(String)
    last_active_at = Column(String)
    updated_at = Column(String)
    created_at = Column(String)
    api_key = Column(String)

Base.metadata.create_all(ow_engine)
OWSession = sessionmaker(bind=ow_engine)
ow_sess = OWSession()
ow_sess.add_all([
    OWUser(id="u1", name="Alice", email="alice@example.com", role="admin", profile_image_url="/x.png", last_active_at="0", updated_at="0", created_at="0", api_key=None),
    OWUser(id="u2", name="Bob", email="bob@example.com", role="user", profile_image_url="/y.png", last_active_at="0", updated_at="0", created_at="0", api_key=None),
])
ow_sess.commit()
ow_sess.close()

# Add an OpenWebUI model table and a couple of models
ModelBase = Base

class OWModel(ModelBase):
    __tablename__ = "model"
    id = Column(String, primary_key=True)
    user_id = Column(String)
    base_model_id = Column(String)
    name = Column(String)
    updated_at = Column(BigInteger)
    created_at = Column(BigInteger)

ModelBase.metadata.create_all(ow_engine)
OWSession2 = sessionmaker(bind=ow_engine)
ow_sess2 = OWSession2()
ow_sess2.add_all([
    OWModel(id="mistral:latest", user_id="u1", base_model_id=None, name="Mistral Latest", updated_at=2, created_at=1),
    OWModel(id="gpt-4o", user_id="u2", base_model_id=None, name="GPT-4o", updated_at=3, created_at=1),
])
ow_sess2.commit()
ow_sess2.close()

# Now import CoSMIC pieces and run migrations (create_all fallback)
from internal.db import Base as CosmicBase, engine as CosmicEngine
from internal.models import User, Config, Service, Statistic

CosmicBase.metadata.create_all(bind=CosmicEngine)

from internal.sync import sync_users
summary = sync_users()
print("SYNC:", summary)

# Verify users inserted
from internal.db import SessionLocal
with SessionLocal() as s:
    users = s.query(User).all()
    print("USERS:", [(u.id, u.email, u.role, u.openweb_id) for u in users])

# Sync LLMs and verify
from internal.sync import sync_llms
from internal.models import LLM
lsum = sync_llms()
print("LLMS SYNC:", lsum)
with SessionLocal() as s:
    llms = s.query(LLM).all()
    print("LLMS:", [(l.id, l.openweb_model_id, l.name) for l in llms])

from internal.openwebui_db import get_latest_model_id
print("LATEST MODEL ID:", get_latest_model_id())

# Clean up temp files
try:
    os.remove(cosmic_path)
    os.remove(ow_path)
except Exception:
    pass
