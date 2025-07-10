@app.post("/detect")
async def detect_api(img_file: UploadFile = File(...)):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    img_path = os.path.join(temp_dir, img_file.filename)

    with open(img_path, "wb") as file:
        file.write(await img_file.read())

    detections = detect_objects(img_path)

    save_detection_log(img_file.filename, detections)

    output_url = f"static/output/{img_file.filename}"
    os.remove(img_path)

    return {
        "filename": img_file.filename,
        "detections": detections,
        "output_image": output_url
    }