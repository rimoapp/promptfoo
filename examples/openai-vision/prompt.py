import base64
import httpx

image_urls = [
    'https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/b/b5/Iridescent.green.sweat.bee1.jpg'
]

def get_image_data(image_url: str) -> str:
    return base64.b64encode(httpx.get(image_url).content).decode("utf-8")

def generate_prompt(context):
    print(context)
    image_url = image_urls[0]
    image_data = get_image_data(image_url)
    return [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an expert at answering questions about images. Please write in a short, concise, and concise manner."
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data,
                },
            },
        ],
    }
]