import os
import requests
from pydantic import BaseModel
from typing import List


class CoachingWriterPipeline:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.id = "coaching_writer_pipeline"
        self.name = "CoSMIC Coaching Writer"
        # Base URL of microservice (internal Docker DNS)
        self.base_url = os.environ.get("COACHING_WRITER_BASE_URL", "http://coaching-writer:8001")
        self.MAX_QUERIES_PER_USER = int(os.environ.get("COACHING_WRITER_MAX_QUERIES", "15"))
        self.user_queries_count = {}

    async def on_startup(self):
        print(f"[CoachingWriterPipeline] startup")

    async def on_shutdown(self):
        print(f"[CoachingWriterPipeline] shutdown")

    def _post(self, endpoint: str, json: dict):
        url = f"{self.base_url}{endpoint}"
        try:
            r = requests.post(url, json=json, timeout=120)
            if r.status_code == 200:
                return r.json()
            return {"error": f"Status {r.status_code}: {r.text}"}
        except Exception as e:
            return {"error": str(e)}

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict):
        # Skip system meta queries similar to existing pipeline practice
        if user_message.startswith("###"): return ""

        user_id = body.get("user", {}).get("id", "anon")
        role = body.get("user", {}).get("role", "user")
        count = self.user_queries_count.get(user_id, 0)
        if role != "admin" and count >= self.MAX_QUERIES_PER_USER:
            return "You have reached the maximum number of coaching queries allowed."
        self.user_queries_count[user_id] = count + 1

        # Allow a simple command prefix to disable RAG or set mode:
        # e.g. /norag rest of question, /mode:critique rest of question
        use_rag = True
        mode = None
        text = user_message.strip()

        if text.lower().startswith("/norag "):
            use_rag = False
            text = text[7:].strip()
        if text.lower().startswith("/mode:"):
            first_space = text.find(" ")
            if first_space > -1:
                mode_token = text[6:first_space]
                text = text[first_space+1:].strip()
                mode = mode_token

        payload = {"query": text, "use_rag": use_rag, "mode": mode}
        data = self._post("/coach/query", payload)

        if "error" in data:
            return f"[CoachingWriterPipeline Error] {data['error']}"
        # Basic formatting back to OpenWebUI
        response = data.get("response", "")
        if mode:
            response = f"(Mode: {mode})\n" + response
        return response


# Example manual test (uncomment to run standalone)
# if __name__ == "__main__":
#     p = CoachingWriterPipeline()
#     body = {"user": {"id": 1, "role": "admin"}}
#     print(p.pipe("/mode:critique Improve clarity of this sentence about regression modeling.", "model", [], body))