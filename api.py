from fastapi import FastAPI
from src.opensi_cosmic import OpenSICoSMIC
from pydantic import BaseModel

app = FastAPI()

# Initialize the OpenSICoSMIC class
opensi_cosmic = OpenSICoSMIC()

class UserMessage(BaseModel):
    user_message: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the OpenSICoSMIC API"}

@app.get("/config")
async def get_config():
    try:
        result = opensi_cosmic.config
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.get("/quit")
async def quit():
    try:
        opensi_cosmic.quit()
        return {"status": "success", "message": "Application is shutting down"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.post("/setup-qa/{user_id}")
async def setup_qa(user_id: str):
    try:
        opensi_cosmic.set_up_qa(str(user_id))
        return {"status": "success", "message": "QA setup completed successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.post("/cosmic")
async def process_cosmic(user_message: UserMessage):
    try:
        result = opensi_cosmic(user_message.user_message)
        return {"status": "success", "result": result[0]}
    except Exception as e:
        return {"status": "error", "message": str(e)}

