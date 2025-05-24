from fastapi import APIRouter, File, UploadFile
from mistralai import Mistral
import io
import base64
from PIL import Image
router = APIRouter(tags=["Get Background Description"])
API_KEY = "S6J7LAS58C518q8Co4PfVZoDjkxAhYMp"



@router.post("/background_description")
def get_background_description(image: UploadFile = File(...)):
    image.file.seek(0)
    image_for_detect = Image.open(
            io.BytesIO(image.file.read())).convert('RGB')
    buffer = io.BytesIO()
    image_for_detect.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
        
        
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{base64_image}"
    message = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Внимательно проанализируй это изображение и определи, есть ли на нем человек или люди. Опиши, что конкретно ты видишь на изображении, но кратко, и на основе этого сделай вывод - да или нет. Если не уверен на 100%, укажи это. Если на изображении есть человек или люди укажи кратко какое у них положение: Стоит/Стоят или Лежит/Лежат. Если получается ситуация когда один человек стоит, а другой лежит, то так и указывай."
            },
            {
                "type": "image_url",
                "image_url": image_url
            }
        ]
    }
    ]
    
    mistral = Mistral(API_KEY)
    response = mistral.chat.complete(
            model="pixtral-12b-2409",
            messages=message,)
    return {"description": response.choices[0].message.content}
