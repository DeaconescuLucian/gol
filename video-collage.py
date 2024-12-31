from PIL import Image
import random
import numpy as np

def create_collage(images):
    if len(images) != 36:
        raise ValueError("Input should contain exactly four images")

    # Create a new blank 1024x1024 pixel image
    collage = Image.new('RGB', (768, 768))

    positions = [(x, y) for x in range(0, 768, 128) for y in range(0, 768, 128)]
    for i in range(36):
        collage.paste(images[i], positions[i])

    return collage

# Example usage:
# if __name__ == "__main__":
#     image_paths = ["image1.png", "image2.png", "image3.png", "image4.png"]
#     images = [Image.open(path) for path in image_paths]
#
#     collage = create_collage(images)
#     collage.show()
#     collage.save("output_collage.png")

# numbers = list(range(9))
# random.shuffle(numbers)
# clips = np.array(numbers)
i = 0
incr = 1
gen = 0
# while gen < 1000:
#     folders = [f"edited-clip/{clip}" for clip in clips]
#     images = [Image.open(f"{folder}/{i}.png") for folder in folders]
#     collage = create_collage(images)
#     collage.save(f"edited-clip-collage/{gen}.png")
#     print(f"Created collage {gen}")
#     i += incr
#     gen += 1
#     if i % 198 == 0:
#         incr *= -1
#
# from createVideo import create
#
# create(f"edited-clip-collage", f"edited-clip-collage")



folders=[]
# for i in range(6):
#     numbers = list(range(6))
#     random.shuffle(numbers)
#     clips = np.array(numbers)
#     folders = folders + [f"135-190-255-05-{clip}" for clip in clips]
#
increment = 0
for i in range(6):
    for j in range(3):
        if (increment + j) % 2 == 0:
            folders = folders + [f"135-190-255-05-1", f"135-190-255-05-4"]
        else:
            folders = folders + [f"135-190-255-05-4",f"135-190-255-05-1"]
    if i % 2 == 0:
        increment += 1
#

while gen < 352:
    images = [Image.open(f"{folder}/{gen}.jpg") for folder in folders]
    collage = create_collage(images)
    collage.save(f"rec/{gen}.png")
    print(f"Created collage {gen}")
    i += incr
    gen += 1

from createVideo import create

create(f"rec", f"135-190-255-05-1+4")

