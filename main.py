from fastapi import FastAPI
from api import cifar_ten, fashion, numbers
import uvicorn


app = FastAPI()
app.include_router(numbers.mnist_router)
app.include_router(fashion.fashion_router)
app.include_router(cifar_ten.cifar_ten_router)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=800)
