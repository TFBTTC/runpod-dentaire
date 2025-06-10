import json
from handler import handler


if __name__ == "__main__":
    event = {
        "image_url": "https://pmvniudsgjeoqcyudpba.supabase.co/storage/v1/object/public/testt//17_png.rf.73d7b07b0dfa5d9c33755f36dfbc173a.jpg"
    }
    result = handler(event)
    print(json.dumps(result, indent=2, ensure_ascii=False))
