import os
import uuid
import base64
from io import BytesIO

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client, handle_file
from pydub import AudioSegment  # used to process and crop the audio

# Create your FastAPI instance
app = FastAPI()

# Define a pydantic model for the request
class EbookRequest(BaseModel):
    ebook_file: str  # base64-encoded string of the text file content

# Constants for file paths
GRADIO_SERVER_URL = "http://0.0.0.0:7860/"
VOICE_FILE_PATH = "/Users/apple/Desktop/projects/untitled_folder/ebook2audiobook/output-1 copy.wav"
AUDIOBOOKS_DIR = "/Users/apple/Desktop/projects/untitled_folder/ebook2audiobook/audiobooks/gui/host/web-5f7080cd-eaf3-4e55-affc-9184a3d0943c/"

@app.post("/convert")
async def convert_ebook(req: EbookRequest):
    # Create a unique temporary filename for the ebook text file.
    tmp_txt_filename = f"tmp_{uuid.uuid4().hex}.txt"
    tmp_txt_path = os.path.join("/tmp", tmp_txt_filename)
    
    try:
        # Decode the base64 ebook file and write it out as a temporary text file.
        ebook_data = base64.b64decode(req.ebook_file)
        with open(tmp_txt_path, "wb") as f:
            f.write(ebook_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding or writing ebook file: {e}")
    
    # Create a Gradio client and call the prediction function.
    try:
        client = Client(GRADIO_SERVER_URL)
        # Use the temporary text file instead of a PDF.
        result = client.predict(
            session="Hello_world!!",
            device="CPU",
            ebook_file=handle_file(tmp_txt_path),
            voice_file=handle_file(VOICE_FILE_PATH),
            language="English",
            custom_model_file="none",
            temperature=0.65,
            length_penalty=1,
            repetition_penalty=2.5,
            top_k=50,
            top_p=0.8,
            speed=1,
            enable_text_splitting=True,
            fine_tuned="DeathPuss&Boots",
            api_name="/submit_convert_btn"
        )
    except Exception as e:
        if os.path.exists(tmp_txt_path):
            os.remove(tmp_txt_path)
        raise HTTPException(status_code=500, detail=f"Error calling Gradio service: {e}")
    
    # Remove the temporary text file after conversion.
    if os.path.exists(tmp_txt_path):
        os.remove(tmp_txt_path)
    
    # Remove the .txt extension from the temporary filename before adding .m4b
    base_name, _ = os.path.splitext(tmp_txt_filename)
    audiobook_filename = f"{base_name}.m4b"  # we name it .m4b even though we'll export in mp4 format
    audiobook_path = os.path.join(AUDIOBOOKS_DIR, audiobook_filename)
    
    if not os.path.exists(audiobook_path):
        audiobook_path = os.path.join("./", audiobook_filename)
        if not os.path.exists(audiobook_path):
            raise HTTPException(status_code=500, detail=f"Audiobook file not found at {audiobook_path}")

    try:
        # Load the audiobook file using pydub.
        audio = AudioSegment.from_file(audiobook_path)
        # Crop out (remove) the first 12 seconds (pydub works in milliseconds).
        cropped_audio = audio[12000:]
        
        # Export the cropped audio into a BytesIO buffer.
        buf = BytesIO()
        # Export as mp4 with AAC codec. Although we export as "mp4", you can still call it .m4b.
        cropped_audio.export(buf, format="mp4", codec="aac")
        cropped_bytes = buf.getvalue()
        
        # Convert the cropped audio bytes to a base64-encoded string.
        audiobook_base64 = base64.b64encode(cropped_bytes).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audiobook file: {e}")
    
    # Optionally, remove the audiobook file after processing:
    # os.remove(audiobook_path)
    
    return {
        "message": f"Result: Audiobook {audiobook_filename} created!",
        "audiobook_base64": audiobook_base64
    }
