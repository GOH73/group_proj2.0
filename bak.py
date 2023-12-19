import cv2
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import aiohttp
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import asyncio
import tkinter as tk
from tkinter import filedialog

heads = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
}


async def async_download_image(session, url, save_path):
    async with session.get(url) as response:
        image = await response.read()
        with open(save_path, 'wb') as f:
            f.write(image)


async def async_download(urls, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        tasks = [async_download_image(session, url, os.path.join(save_dir, f'image_{i}.jpg')) for i, url in
                 enumerate(urls, start=1)]
        await asyncio.gather(*tasks)


def get_picture(page, page_size):
    urls = []
    url = (f"https://www.logosc.cn/api/so/get?category=pixabay&isNeedTranslate="
           f"false&keywords=%E5%8A%A8%E7%89%A9&page={page}&pageSize={page_size}")
    print(url)
    response = requests.get(url=url, headers=heads)
    content = response.json()
    if "data" in content:
        i = 0
        while True:
            try:
                if content["data"][i]["large_img_path"]["url"]:
                    picture_url = content["data"][i]["large_img_path"]["url"]
                    print("picture_url" + str(i) + ":", picture_url)
                    i += 1
                    urls.append(picture_url)
            except:
                print("没有数据！")
                break
    else:
        print("没有获取到数据！")
    return urls


def download_image(url, save_path):
    image = requests.get(url).content
    with open(save_path, 'wb') as f:
        f.write(image)


def download(urls, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    with ProcessPoolExecutor() as executor:
        for i, url in enumerate(urls, start=1):
            image_name = f'image_{i}.jpg'
            image_path = os.path.join(save_dir, image_name)
            executor.submit(download_image, url, image_path)
            print(f'{image_name} 正在保存。。。')


def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


def compute_similarity(query_features, image_features):
    query_features = np.vstack(query_features).reshape(1, -1)
    image_features = np.vstack(image_features).reshape(1, -1)
    max_dim = max(query_features.shape[1], image_features.shape[1])
    query_features = np.pad(query_features, ((0, 0), (0, max_dim - query_features.shape[1])), mode='constant')
    image_features = np.pad(image_features, ((0, 0), (0, max_dim - image_features.shape[1])), mode='constant')
    similarity = cosine_similarity(query_features, image_features)
    return similarity[0][0]


def parallel_extract_features(image_paths):
    with ProcessPoolExecutor() as executor:
        features = list(executor.map(extract_features, image_paths))
    return features


async def spider_and_compute_similarity(query_image_path):
    save_dir = 'images'
    # urls = get_picture(1, 200)

    # 使用asyncio异步下载图像
    # await async_download(urls, save_dir)

    # 尝试从文件加载特征
    features_filename = 'features.pkl'
    if os.path.exists(features_filename):
        with open(features_filename, 'rb') as file:
            features = pickle.load(file)
        image_paths = [os.path.join(save_dir, image_name) for image_name in os.listdir(save_dir)]
    else:
        # 从未保存的文件中提取特征
        image_paths = [os.path.join(save_dir, image_name) for image_name in os.listdir(save_dir)]
        features = parallel_extract_features(image_paths)
        # 保存特征到文件
        with open(features_filename, 'wb') as file:
            pickle.dump(features, file)

    query_features = extract_features(query_image_path)

    similarities = [compute_similarity(query_features, feature) for feature in features]

    similarities_and_paths = list(zip(similarities, image_paths))
    similarities_and_paths.sort(key=lambda x: x[0], reverse=True)

    return similarities_and_paths


def browse_button():
    global query_image_path
    query_image_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select an Image",
                                                   filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
    entry_path.delete(0, tk.END)
    entry_path.insert(0, query_image_path)


def search_images():
    if not entry_path.get():
        return

    query_image_path = entry_path.get()
    results = asyncio.run(spider_and_compute_similarity(query_image_path))

    num_similar_images = 10
    plt.figure(figsize=(12, 6))
    for i, (similarity, image_path) in enumerate(results[:num_similar_images]):
        image = cv2.imread(image_path)
        plt.subplot(2, 5, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Similarity: {similarity:.2f}')
        plt.axis('off')

    plt.show()


if __name__ == '__main__':
    # Set up Tkinter GUI
    root = tk.Tk()
    root.title("图像搜索引擎")

    # Create and place widgets
    label_path = tk.Label(root, text="查询图像路径:")
    label_path.grid(row=0, column=0, padx=10, pady=5)

    entry_path = tk.Entry(root, width=50)
    entry_path.grid(row=0, column=1, padx=10, pady=5)

    browse_button = tk.Button(root, text="浏览", command=browse_button)
    browse_button.grid(row=0, column=2, padx=10, pady=5)

    search_button = tk.Button(root, text="搜图", command=search_images)
    search_button.grid(row=1, column=0, columnspan=3, pady=10)

    # Run Tkinter main loop
    root.mainloop()
