from insightface.app import FaceAnalysis
import cv2

# 初始化 buffalo_l
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))  # det_size 可调

img = cv2.imread("/Users/shenqi/Documents/code/python/SeiyuuMatch/tests/DD64ED12CC1384156BAABC14297FA42A.png")
faces = app.get(img)

print(f"检测到 {len(faces)} 张人脸")

for face in faces:
    emb = face.normed_embedding  # 512-d embedding
    quality = face.det_score      # 可当作粗略质量参考
    print("embedding:", emb[:5], "...", "score:", quality)