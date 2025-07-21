from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Configuration ESERISIA AI
app = FastAPI(
    title="eserisia-api-ultimate API",
    description="API générée par ESERISIA AI - L'IDE le plus puissant",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    price: float

class ItemResponse(BaseModel):
    message: str
    data: Optional[Item] = None

# In-memory storage (remplacer par PostgreSQL)
items_db: List[Item] = []

@app.get("/")
async def root():
    return {
        "message": "🚀 eserisia-api-ultimate API",
        "powered_by": "ESERISIA AI",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/items", response_model=List[Item])
async def get_items():
    return items_db

@app.post("/items", response_model=ItemResponse)
async def create_item(item: Item):
    item.id = len(items_db) + 1
    items_db.append(item)
    return ItemResponse(message="Item créé avec succès", data=item)

@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    for item in items_db:
        if item.id == item_id:
            return ItemResponse(message="Item trouvé", data=item)
    raise HTTPException(status_code=404, detail="Item non trouvé")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
