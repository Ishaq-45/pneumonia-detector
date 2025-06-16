# # # import numpy as np
# # # from tensorflow.keras.models import load_model
# # # from tensorflow.keras.preprocessing import image
# # # import matplotlib.pyplot as plt

# # # # === 1. Load the trained model ===
# # # model = load_model('pneumonia_model.h5')

# # # # === 2. Load and preprocess image ===
# # # img_path = 'sample.jpg'  # <-- Change this to your image path
# # # img = image.load_img(img_path, target_size=(150, 150))
# # # img_array = image.img_to_array(img)
# # # img_array = np.expand_dims(img_array, axis=0)
# # # img_array /= 255.0  # Normalize

# # # # === 3. Make prediction ===
# # # prediction = model.predict(img_array)

# # # # === 4. Display result ===
# # # plt.imshow(img)
# # # plt.axis('off')
# # # if prediction[0][0] > 0.5:
# # #     plt.title("🔴 Prediction: Pneumonia")
# # #     print("🔴 Predicted: Pneumonia")
# # # else:
# # #     plt.title("🟢 Prediction: Normal")
# # #     print("🟢 Predicted: Normal")
# # # plt.show()
# # import numpy as np
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing import image
# # import matplotlib.pyplot as plt
# # import os

# # # === 1. Load the trained model ===
# # model = load_model('pneumonia_model.h5')

# # # === 2. Define image paths ===
# # # You can add more image paths to this list
# # # image_paths = [
# # #     'sample.jpg','result1.jpg','result2.jpg'  # Replace/add paths as needed
# # #     # 'dataset/chest_xray/test/NORMAL1.jpg',
# # #     # 'dataset/chest_xray/test/PNEUMONIA2.jpg'
# # # ]
# # image_paths = [f for f in os.listdir() if f.endswith(('.jpg', '.png'))]
# # print("Images found:", image_paths)

# # print("Current working directory:", os.getcwd())

# # # === 3. Process and predict each image ===
# # for idx, img_path in enumerate(image_paths):
# #     try:
# #         img = image.load_img(img_path, target_size=(150, 150))
# #         img_array = image.img_to_array(img)
# #         img_array = np.expand_dims(img_array, axis=0)
# #         img_array /= 255.0  # Normalize

# #         prediction = model.predict(img_array)
# #         confidence = prediction[0][0]

# #         # Display and save result
# #         plt.imshow(img)
# #         plt.axis('off')

# #         if confidence > 0.5:
# #             title = f"Prediction: Pneumonia ({confidence*100:.2f}%)"
# #             print(f"🔴 {img_path} => Pneumonia ({confidence*100:.2f}%)")
# #         else:
# #             title = f"Prediction: Normal ({(1 - confidence)*100:.2f}%)"
# #             print(f"🟢 {img_path} => Normal ({(1 - confidence)*100:.2f}%)")

# #         plt.title(title)
# #         result_path = f'result_{idx+1}.png'
# #         plt.savefig(result_path)
# #         print(f"✅ Saved: {result_path}")
# #         plt.show()

# #     except Exception as e:
# #         print(f"⚠️ Error processing {img_path}: {e}")
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt
# import os

# # === 1. Load the trained model ===
# model = load_model('pneumonia_model.h5')

# # === 2. Auto-detect image files ===
# image_paths = [f for f in os.listdir() if f.endswith(('.jpg', '.png'))]
# print("📝 Images found:", image_paths)

# # === 3. Process and predict each image ===
# for idx, img_path in enumerate(image_paths):
#     try:
#         img = image.load_img(img_path, target_size=(150, 150))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array /= 255.0  # Normalize

#         prediction = model.predict(img_array)
#         confidence = prediction[0][0]

#         # Display image and prediction
#         plt.imshow(img)
#         plt.axis('off')

#         if confidence > 0.5:
#             title = f"🔴 Prediction: Pneumonia ({confidence*100:.2f}%)"
#             print(f"🔴 {img_path} => Pneumonia ({confidence*100:.2f}%)")
#         else:
#             title = f"🟢 Prediction: Normal ({(1 - confidence)*100:.2f}%)"
#             print(f"🟢 {img_path} => Normal ({(1 - confidence)*100:.2f}%)")

#         plt.title(title)

#         # Save the result image
#         result_path = f'result_{idx+1}.png'
#         plt.savefig(result_path)
#         print(f"✅ Saved: {result_path}")

#         plt.show()

#     except Exception as e:
#         print(f"⚠️ Error processing {img_path}: {e}")


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import shutil

# === 1. Load the trained model ===
model = load_model('pneumonia_model.h5')

# === 2. Auto-detect image files ===
image_paths = [f for f in os.listdir() if f.endswith(('.jpg', '.png'))]
print("📝 Images found:", image_paths)

# === 3. Prepare processed_images folder ===
processed_folder = 'processed_images'
os.makedirs(processed_folder, exist_ok=True)

# === 4. Prepare CSV file ===
csv_filename = 'predictions_report.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_name', 'prediction', 'confidence_percent', 'timestamp'])

    # === 5. Process each image ===
    for idx, img_path in enumerate(image_paths):
        try:
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize

            prediction = model.predict(img_array)
            confidence = prediction[0][0]

            plt.imshow(img)
            plt.axis('off')

            if confidence > 0.5:
                pred_label = 'Pneumonia'
                conf_percent = confidence * 100
                title = f"🔴 Prediction: {pred_label} ({conf_percent:.2f}%)"
                print(f"🔴 {img_path} => {pred_label} ({conf_percent:.2f}%)")
            else:
                pred_label = 'Normal'
                conf_percent = (1 - confidence) * 100
                title = f"🟢 Prediction: {pred_label} ({conf_percent:.2f}%)"
                print(f"🟢 {img_path} => {pred_label} ({conf_percent:.2f}%)")

            plt.title(title)

            result_path = f'result_{idx + 1}.png'
            plt.savefig(result_path)
            print(f"✅ Saved: {result_path}")
            plt.close()

            # Record timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Write to CSV
            writer.writerow([img_path, pred_label, f"{conf_percent:.2f}", timestamp])

            # Move original image to processed_images folder
            shutil.move(img_path, os.path.join(processed_folder, img_path))

        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")

print(f"📁 All predictions saved to: {csv_filename}")
print(f"📂 Processed images moved to: {processed_folder}")
