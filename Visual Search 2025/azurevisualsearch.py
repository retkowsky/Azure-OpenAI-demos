# Azure VisualSearch
# Serge Retkowsky | serge.retkowsky@microsoft.com
# Updates: 29-07-2025


# Librairies
import base64
import datetime
import io
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import re
import requests
import seaborn as sns
import threading

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    VectorSearch,
)
from azure.storage.blob import BlobServiceClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from io import BytesIO
from IPython.display import display, HTML
from PIL import Image
from queue import Queue
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Variables
load_dotenv("azure.env")
# Azure AI Vision
azure_vision_key = os.getenv("azure_vision_key")
azure_vision_endpoint = os.getenv("azure_vision_endpoint")
# Azure AI Search
azure_search_endpoint = os.getenv("azure_search_endpoint")
azure_search_key = os.getenv("azure_search_key")
# Azure storage account
blob_connection_string = os.getenv("blob_connection_string")
container_name = os.getenv("container_name")
blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
container_client = blob_service_client.get_container_client(container_name)


# Dirs
IMAGES_DIR = "images"
IMAGESGEN_DIR= f"{IMAGES_DIR}/ai_genereated_images"
os.makedirs(IMAGESGEN_DIR, exist_ok=True)


# Helper

## 1. Utilities

def download_image(image_filename: str, output_filename: str) -> None:
    """
    Downloads an image blob from an Azure Blob Storage container and saves it locally.

    Args:
        container_name (str): Name of the Azure Blob Storage container.
        image_filename (str): The name of the blob (image file) to download.
        output_filename (str): Local filename to save the downloaded image.
        connection_string (str): Azure Blob Storage connection string.

    Raises:
        Exception: If the image download or save fails.
    """
    try:
        # Connect to the blob service
        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=image_filename)

        # Download the image data
        download_stream = blob_client.download_blob()
        image_data = io.BytesIO(download_stream.readall())

        # Load and save the image
        with Image.open(image_data) as img:
            img.save(output_filename)
        
        print(f"✅ Image successfully downloaded from '{container_name}/{image_filename}' and saved to '{output_filename}'.")

    except Exception as e:
        print(f"❌ Failed to download and save image: {e}")
        raise


def display_image(container_client:str, image_file: str) -> None:
    """
    Downloads an image from Azure Blob Storage and displays it.

    Args:
        container_client (azure.storage.blob.ContainerClient): The container client to access the blob storage.
        image_file (str): The name of the image file to be downloaded and viewed.

    Returns:
        None
    """
    try:
        blob_client = container_client.get_blob_client(image_file)
        blob_image = container_client.get_blob_client(image_file).download_blob().readall()
    
        # Create an in-memory stream
        image_stream = io.BytesIO(blob_image)
    
        # Open and display the image
        print(f"🖼️ Image: {image_file}")
        image = Image.open(image_stream)
        image.thumbnail((640, 640), Image.LANCZOS)
        display(image)

    except Exception as e:
        print(f"❌ Failed to load/display image '{image_file}': {e}")


def display_images(image_list:str, images_per_row: int = 3) -> None:
    """
    Displays a grid of images using matplotlib with automatic layout and titles.

    Parameters:
    -----------
    image_list : List[Union[str, PIL.Image.Image]]
        A list of image file paths or PIL Image objects to display.
    images_per_row : int, optional
        Number of images to display per row (default is 3).

    Returns:
    --------
    None
        Displays the image grid inline.
    """
    n_images = len(image_list)
    n_rows = math.ceil(n_images / images_per_row)
    figsize_per_image = (4, 4)

    fig, axes = plt.subplots(
        n_rows,
        images_per_row,
        figsize=(figsize_per_image[0] * images_per_row, figsize_per_image[1] * n_rows)
    )

    # Normalize axes to a flat list
    if n_rows == 1:
        axes = [axes] if images_per_row == 1 else axes
    axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

    for i, image in enumerate(image_list):
        try:
            if isinstance(image, str):
                img = Image.open(image)
                title = os.path.basename(image)
            elif isinstance(image, Image.Image):
                img = image
                title = "Image"
            else:
                raise ValueError("Unsupported image type.")

            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(title, fontsize=8)
        except Exception as e:
            axes[i].axis('off')
            axes[i].set_title(f"Error: {e}", fontsize=8)

    # Hide any unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def get_cosine_similarity(vector1: Sequence[float], vector2: Sequence[float]) -> float:
    """
    Computes the cosine similarity between two vectors.

    Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space 
    that measures the cosine of the angle between them.

    Args:
        vector1 (array-like): The first vector.
        vector2 (array-like): The second vector.

    Returns:
        float: The cosine similarity between the two vectors, ranging from -1 to 1.
    """
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cosine similarity is undefined for zero-magnitude vectors.")

    similarity = np.dot(vector1, vector2) / (norm1 * norm2)
    
    return similarity


def plot_2D_umap(RESULTS_DIR:str, proj_2d:str, catalog_df: pd.DataFrame, category:str) -> None:
    """
    Plots a 2D UMAP projection of image vectors embeddings and saves the plot as an image file.

    Args:
        category (str): The name of the categorical variable to color the points in the scatter plot.

    Returns:
        None
    """
    try:
        fig_2d = px.scatter(
            proj_2d,
            x=0,
            y=1,
            color=catalog_df[category].tolist(),
            labels={"color": category},
            custom_data=[catalog_df["imagefile"].tolist()],
            title="Images Vectors Embeddings UMAP Projections",
            height=640,
        )
        fig_2d.update_traces(marker_size=3, hovertemplate="%{customdata}")
        fig_2d.show()
        
        # Saving imagefile
        output_file = f"2D_{category}.png"
        fig_2d.write_image(os.path.join(RESULTS_DIR, output_file))
    
        return output_file

    except Exception as e:
        print(f"❌ Failed to generate or save UMAP plot: {e}")
        return None

    
def plot_3D_umap(RESULTS_DIR:str, proj_3d:str, catalog_df: pd.DataFrame, category:str) -> None:
    """
    Plots a 3D UMAP projection of image vectors embeddings and saves the plot as an image file.

    Args:
        category (str): The name of the categorical variable to color the points in the scatter plot.

    Returns:
        None
    """
    try:
        fig_3d = px.scatter_3d(
            proj_3d,
            x=0,
            y=1,
            z=2,
            color=catalog_df[category].tolist(),
            labels={"color": category},
            custom_data=[catalog_df["imagefile"].tolist()],
            title="Images Vectors Embeddings UMAP Projections",
            height=860,
        )
        fig_3d.update_traces(marker_size=2, hovertemplate="%{customdata}")
        fig_3d.show()
        
        # Saving imagefile
        output_file = f"3D_{category}.png"
        fig_3d.write_image(os.path.join(RESULTS_DIR, output_file))
    
        return output_file

    except Exception as e:
        print(f"❌ Failed to generate or save UMAP plot: {e}")
        return None


def plot_categories(catalog_df: pd.DataFrame, variable: str, palette: str = "Set2") -> None:
    """
    Plot the distribution of categories in a specified column of a DataFrame.

    Generates a bar plot using Seaborn, showing the count of each unique category in the given column.
    Includes count labels on bars, custom styling, and improved layout.

    Parameters
    ----------
    catalog_df : pd.DataFrame
        The DataFrame containing the data.
    variable : str
        The name of the categorical column to plot.
    palette : str, optional
        The name of the Seaborn color palette to use. Default is "Set2".

    Returns
    -------
    None
        Displays the plot.
    """
    if variable not in catalog_df.columns:
        raise ValueError(f"Column '{variable}' not found in DataFrame.")

    if catalog_df[variable].isnull().all():
        raise ValueError(f"Column '{variable}' contains only NaN values.")

    category_counts = catalog_df[variable].value_counts().sort_values(ascending=False)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    ax = sns.countplot(
        data=catalog_df,
        x=variable,
        hue=variable,
        palette=palette,
        order=category_counts.index,
        legend=False
    )

    ax.set_xlabel(variable, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Distribution of '{variable}' Categories", fontsize=14, weight='bold')
    plt.xticks(rotation=45, ha='right')

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}',
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom',
                    fontsize=10, xytext=(0, 3),
                    textcoords='offset points')

    plt.tight_layout()
    plt.show()


def view_search_results(results:list) -> None:
    """
    Displays a visual HTML summary of image classification or ranking results.

    This function takes a list of tuples containing image filenames and their associated scores,
    retrieves the corresponding images from an Azure Blob Storage container, resizes them for
    display, and renders them in a styled HTML layout using base64-encoded inline images.

    Parameters:
    -----------
    results : list of tuples
        A list where each element is a tuple of the form (filename: str, score: float),
        representing the image filename in the blob storage and its associated score.

    Behavior:
    ---------
    - Downloads each image from the Azure Blob Storage using the `container_client`.
    - Resizes each image to 25% of its original dimensions.
    - Encodes the image in base64 and embeds it in an HTML layout.
    - Displays the images in a responsive grid with their rank, filename, and score.
    - If an image fails to load or process, logs a warning message to the console.

    Returns:
    --------
    None
    """
    html_content = '<div style="display:flex; flex-wrap: wrap; gap: 10px;">'

    for idx, (filename, score) in enumerate(results, start=1):
        try:
            blob_client = container_client.get_blob_client(filename)
            image_data = blob_client.download_blob().readall()
            
            with Image.open(io.BytesIO(image_data)) as img:
                original_width, original_height = img.size
                resized_img = img.resize((original_width // 4, original_height // 4))
                buffered = io.BytesIO()
                resized_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
    
                html_content += f'''
                <div style="flex: 0 0 calc(33.33% - 10px); text-align:center;">
                    <div style="
                        font-weight: bold; 
                        margin-bottom: 6px; 
                        background-color: #f0f0f0; 
                        padding: 6px 8px; 
                        border-radius: 6px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    ">
                        Top {idx}<br/>
                        Image: {filename}<br/>
                        Score = {score:.4f}
                    </div>
                    <img src="data:image/png;base64,{img_str}" style="max-width:100%; height:auto; border-radius: 8px;"/>
                </div>
                '''
        except Exception as e:
            print(f"⚠️ Failed to load/display image {filename}: {e}")

    html_content += '</div>'
    display(HTML(html_content))


## 2. Azure AI Vision

def describe_image(container_client: str, image_file: str) -> None:
    """
    Analyzes an image using Azure AI Vision to generate tags and captions from an image.

    Args:
        container_client (azure.storage.blob.ContainerClient): The container client to access the blob storage.
        image_file (str): The name of the image file to be downloaded and analyzed.

    Returns:
        None
    """
    api_version = "2024-02-01"
    options = "&features=tags,caption"
    
    model = f"?api-version={api_version}&model-version=2023-10-01"
    url = f"{azure_vision_endpoint}computervision/imageanalysis:analyze{model}{options}"
    
    # Header
    headers_cv = {
        "Content-type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": azure_vision_key,
    }
    
    # Image
    blob_client = container_client.get_blob_client(image_file)
    blob_image = blob_client.download_blob().readall()
    image_stream = io.BytesIO(blob_image)
    image = Image.open(image_stream)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    response = requests.post(url, data=image_bytes, headers=headers_cv)

    if response.status_code == 200:
        results = response.json()
        print("🧠 Automatic analysis of the image using Azure AI Vision:")
        print("\033[1;31;34m")
        print("📝 Main caption:")
        caption = results["captionResult"]["text"]
        confidence = results["captionResult"]["confidence"]
        print(f"💬 {caption} = {confidence:.3f}")
        print("\033[1;31;32m")
        print("Detected tags:")
        tags = results["tagsResult"]["values"]

        for tag in tags:
            name = tag["name"]
            confidence = tag["confidence"]
            print(f"🔖 {name} = {confidence:.5f}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None


def text_embedding(prompt: str) -> [List[float]]:
    """
    Vectorizes a given text using Azure AI Vision's text vectorization feature.

    Args:
        prompt (str): The text to be vectorized.

    Returns:
        list or None: The vector representation of the text if the request is successful, otherwise None.
    """
    api_version = "2024-02-01"
    model_version = "2023-04-15"

    version = f"?api-version={api_version}&model-version={model_version}"
    vec_txt_url = f"{azure_vision_endpoint}computervision/retrieval:vectorizeText{version}"
    headers = {
        "Content-type": "application/json",
        "Ocp-Apim-Subscription-Key": azure_vision_key
    }
    payload = {"text": prompt}
    response = requests.post(vec_txt_url, json=payload, headers=headers)

    if response.status_code == 200:
        text_emb = response.json().get("vector")
        return text_emb

    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None


session = requests.Session()
def image_embedding(imagefile: str) -> Optional[List[float]]:
    """
    Embeds an image using Azure AI Vision's image vectorization feature.

    Args:
        imagefile (str): The name of the image file to be vectorized.

    Returns:
        list or None: The vector representation of the image if the request is successful, otherwise None.
    """
    api_version = "2024-02-01"
    model_version = "2023-04-15"

    version = f"?api-version={api_version}&model-version={model_version}"
    vec_img_url = f"{azure_vision_endpoint}computervision/retrieval:vectorizeImage{version}"
    headers = {
        "Content-type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": azure_vision_key,
    }
    
    try:
        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(imagefile)
        
        stream = BytesIO()
        blob_data = blob_client.download_blob()
        blob_data.readinto(stream)
        stream.seek(0)  # Reset stream position to the beginning
        response = session.post(vec_img_url, data=stream, headers=headers)
        response.raise_for_status()  # Raise an exception if response is not 200
        image_emb = response.json()["vector"]

        return image_emb

    except requests.exceptions.RequestException as e:
        print(f"❌ Request Exception: {e}")
    except Exception as ex:
        print(f"❌ Error: {ex}")

    return None


def get_image_embedding(local_image: str) -> List[float]:
    """
    Generates vector embeddings for a local image using Azure AI Vision.

    Parameters:
    local_image (str): The file path to the local image.

    Returns:
    list: A list of vector embeddings representing the image.
    """
    api_version = "2024-02-01"
    model_version = "2023-04-15"
    
    headers = {
        "Content-type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": azure_vision_key,
    }
    version = f"?api-version={api_version}&model-version={model_version}"
    vec_img_url = f"{azure_vision_endpoint}computervision/retrieval:vectorizeImage{version}"

    try:
        # Reading the images in binary
        with open(local_image, "rb") as f:
            data = f.read()

        # Sending the request
        response = requests.post(vec_img_url, data=data, headers=headers)

        # Get the vector embeddings
        image_emb = response.json()["vector"]

        return image_emb

    except Exception as e:
        print(f"❌ Failed to vectorize image '{local_image}': {e}")
        raise


## 3. Azure AI Search

def delete_index(index_name: str) -> None:
    """
    Deletes an Azure AI Search index.

    Args:
        index_name (str): The name of the index to be deleted.

    Returns:
        None
    """   
    try:
        search_client = SearchIndexClient(endpoint=azure_search_endpoint, credential=AzureKeyCredential(azure_search_key))
        print(f"🧹 Deleting the Azure AI Search index: {index_name}")
        search_client.delete_index(index_name)
        print("Done")

    except Exception as e:
        print(f"❌ Failed to delete index '{index_name}': {e}")


def get_index_stats(index_name: str) -> Tuple[int, int]:
    """
    Get statistics about Azure AI Search index

    Args:
        index_name (str): The name of the index whose status is to be retrieved.

    Returns:
        Results tuple
    """  
    url = f"{azure_search_endpoint}/indexes/{index_name}/stats?api-version=2024-07-01"
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_search_key,
    }
    response = requests.get(url, headers=headers)
    print(f"📡 Azure AI Search index status for: {index_name}\n")

    if response.status_code == 200:
        res = response.json()
        print(json.dumps(res, indent=2))
        document_count = res['documentCount']
        storage_size = res['storageSize']

    else:
        print(f"❌ Request failed with status code: {response.status_code}")
    
    return document_count, storage_size


def get_index_status(index_name: str) -> None:
    """
    Retrieves and prints the status of an Azure AI Search index.

    Args:
        index_name (str): The name of the index whose status is to be retrieved.

    Returns:
        None
    """
    print(f"Azure AI Search Index: {index_name}\n")

    headers = {"Content-Type": "application/json", "api-key": azure_search_key}
    params = {"api-version": "2024-07-01"}

    index_status = requests.get(
        f"{azure_search_endpoint}/indexes/{index_name}",
        headers=headers,
        params=params)
    
    try:
        print(json.dumps((index_status.json()), indent=5))
    except:
        print("❌ Request failed")


def process_single_image(item, progress_queue):
    """
    Processes a single image by generating its embedding vector and updating the item dictionary.

    Parameters:
    - item (dict): A dictionary containing image metadata. Expected keys:
        - "idfile": Identifier for the image file.
        - "imagefile": Path or reference to the image file.
    - progress_queue (Queue): A multiprocessing or threading queue used to signal progress.

    Returns:
    - dict: The updated item dictionary with an added "imagevector" key containing the image embedding.
    
    Side Effects:
    - Puts a signal (integer 1) into the progress_queue to indicate that processing is complete.
    """
    imgindex = item["idfile"]
    imgfile = item["imagefile"]
    item["imagevector"] = image_embedding(imgfile)
    progress_queue.put(1)  # Signal completion
    
    return item


def progress_report(progress_queue, total_images):
    """
    Monitors and reports the progress of image processing in a background thread.

    Parameters:
    - progress_queue (Queue): A queue used to receive progress signals (typically from worker threads or processes).
    - total_images (int): The total number of images expected to be processed.

    Behavior:
    - Continuously listens for progress signals from the queue.
    - Increments the count of processed images upon receiving each signal.
    - Periodically prints progress updates (every 1000 images or upon completion), including:
        - Timestamp of the report
        - Number of processed images
        - Percentage completed

    Notes:
    - Uses a timeout of 1 second when waiting for queue items to avoid blocking indefinitely.
    - Silently continues on timeout or other exceptions.
    """
    processed = 0
    
    while processed < total_images:
        try:
            progress_queue.get(timeout=1)
            processed += 1
            if processed % 1000 == 0 or processed == total_images:
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                pctdone = round(processed / total_images * 100)
                print(f"✅ {now} Number of processed image files = {processed:06} of {total_images:06} | Done: {pctdone}%")
        except:
            continue


def distribute_image_embedding(max_workers, list_of_files):
    """
    Distributes the task of generating image embeddings across multiple threads.

    Parameters:
    - max_workers (int, optional): The number of threads to use for concurrent processing. Defaults to 2.
    - list_of_files (list): List of files in a json structure
    
    Behavior:
    - Loads image metadata from a JSON file (`json_images_file`).
    - Initializes a progress queue and starts a background thread to report progress.
    - Uses a ThreadPoolExecutor to process images concurrently, where each image is embedded using `process_single_image`.
    - Collects and stores the updated image data in the original order.
    - Waits briefly for the progress thread to finish reporting.

    Returns:
    - List[dict]: A list of updated image metadata dictionaries, each containing the computed image embedding.

    Side Effects:
    - Prints progress updates and status messages to the console.
    - Handles and logs exceptions raised during image processing.
    """
    print("🏃 Running the image files embeddings with threading...")
    
    with open(list_of_files, "r", encoding="utf-8") as file:
        input_data = json.load(file)
    
    image_count = len(input_data)
    print(f"- Total number of images to embed = {image_count}")
    print(f"- Using {max_workers} threads\n")
    
    progress_queue = Queue()
    
    # Start progress reporter thread
    progress_thread = threading.Thread(
        target=progress_report, 
        args=(progress_queue, image_count),
        daemon=True
    )
    progress_thread.start()
    
    # Process images concurrently
    updated_data = [None] * image_count
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_image, item.copy(), progress_queue): idx 
            for idx, item in enumerate(input_data)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                updated_data[idx] = result
            except Exception as exc:
                print(f'Image {idx} generated an exception: {exc}')
    
    progress_thread.join(timeout=1)
    
    return updated_data


def upload_documents(index_name:str, documents: List[Dict]) -> int:
    """
    Loads documents into an Azure AI Search index.

    Args:
        documents (list): A list of documents to be uploaded to the Azure AI Search index.

    Returns:
        int: The number of documents uploaded.
    """
    # Upload some documents to the index
    print(f"📤 Uploading the documents into the index {index_name}...")
    nb_loaded_docs = 0
    
    try:
        # Setting the Azure AI Search client
        search_client = SearchClient(endpoint=azure_search_endpoint, index_name=index_name, credential=AzureKeyCredential(azure_search_key))
        
        # Response
        response = search_client.upload_documents(documents)
        nb_loaded_docs += len(documents)
        print(f"✅ Uploaded {nb_loaded_docs} documents into the Azure AI Search index. Please wait.\n")
        return len(documents)

    except Exception as e:
        print(f"❌ Failed to upload documents to index '{index_name}': {e}")
        return 0


## 4. Visual search

def visual_search_image(index_name:str, query_image: str, topn: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a visual search based on the provided image and returns the top N results.

    Args:
        query_image (str): The image to be vectorized and used for the visual search.
        topn (int, optional): The number of top results to return. Defaults to 5.

    Returns:
        list: A list of tuples containing the filenames and their corresponding cosine similarity scores.
    """    
    try:
        search_client = SearchClient(endpoint=azure_search_endpoint, index_name=index_name, credential=AzureKeyCredential(azure_search_key))
    
        query_vector = get_image_embedding(query_image)
        request = search_client.search(None, vector_queries=[
            VectorizedQuery(vector=query_vector, k_nearest_neighbors=topn, fields="imagevector")])

        # Assuming the search results include cosine similarity scores
        results = [(doc["imagefile"], doc["@search.score"]) for doc in request]

        print("\033[1;34m")
        print("🔍 Top Image Results (Sorted by Cosine Similarity):\n")
        for idx, (filename, score) in enumerate(results, start=1):
            print(f"✅ Top {idx:02}: {filename:<20} | 🔗 Cosine Similarity = {score:.5f}")
        print("\033[0m")

        return results

    except Exception as e:
        print(f"❌ Error during vector search: {e}")
        return []


def visual_search_text(index_name:str, text: str, topn: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a visual search based on the provided text and returns the top N results.

    Args:
        text (str): The text to be vectorized and used for the visual search.
        topn (int, optional): The number of top results to return. Defaults to 5.

    Returns:
        list: A list of tuples containing the filenames and their corresponding cosine similarity scores.
    """   
    try:
        search_client = SearchClient(endpoint=azure_search_endpoint, index_name=index_name, credential=AzureKeyCredential(azure_search_key))
    
        query_vector = text_embedding(text)
        request = search_client.search(None, vector_queries=[
            VectorizedQuery(vector=query_vector, k_nearest_neighbors=topn, fields="imagevector")])

        # Assuming the search results include cosine similarity scores
        results = [(doc["imagefile"], doc["@search.score"]) for doc in request]

        print("\033[1;34m")
        print("🔍 Top Image Results (Sorted by Cosine Similarity):\n")
        for idx, (filename, score) in enumerate(results, start=1):
            print(f"✅ Top {idx:02}: {filename:<20} | 🔗 Cosine Similarity = {score:.5f}")
        print("\033[0m")

        return results
    
    except Exception as e:
        print(f"❌ Error during vector search: {e}")
        return []


## 5. Azure AI Foundry gpt-image-1

def imagen_gptimage1(source_images: List[str], prompt: str, n: int = 1, size: str = "1024x1536") -> None:
    """
    Composes a new image using multiple input images and a prompt via Azure OpenAI's image composition endpoint.

    Parameters:
        source_images (list[str]): List of file paths to the source images to be used for composition.
        prompt (str): Text prompt guiding the composition of the final image.
        n (int, optional): Number of composed images to generate. Defaults to 1.

    Returns:
        list[str] or None: A list of file paths to the saved composed images if successful,
                           or None if an error occurs during processing.

    The function:
        - Sends a multipart request with multiple source images and a prompt to the Azure image composition endpoint.
        - Parses and decodes the base64-encoded images returned in the response.
        - Displays and saves each composed image to a predefined results directory.
        - Returns a list of file paths to the generated images.
    """
    try:
        files = [("image[]", open(image, "rb")) for image in source_images]

        headers = {"api-key": os.getenv("key")}
        aoai_name = re.search(r'https://(.*?)/openai', os.getenv("endpoint")).group(1)
        url = f"https://{aoai_name}/openai/deployments/gpt-image-1/images/edits?api-version=2025-04-01-preview"
        
        data = {
            "prompt": prompt,
            "n": n,
            "size": size,  # Options: 1024x1024, 1536x1024, 1024x1536
            "quality": "high",  # high, medium, low
            "output_compression": 100,
            "output_format": "jpeg",
        }

        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()

        images_data = response.json()["data"]
        encoded_images = [img["b64_json"] for img in images_data]

        # Parsing the generated images
        output_images_list = []

        for encoded_image in encoded_images:
            img = Image.open(BytesIO(base64.b64decode(encoded_image)))
            # Saving image to a file
            now = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3])
            output_file = os.path.join(IMAGESGEN_DIR, f"compose_{now}.jpg")
            img.save(output_file)
            print(f"✅ Generated AI image file is saved: {output_file}")
            output_images_list.append(output_file)
    
        print()
    
    except Exception as e:
        print(f"❌ Error generating images: {e}")
        return None

    return output_images_list

