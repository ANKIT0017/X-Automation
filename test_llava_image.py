import sys
from tweet_bulk import llava_image_understanding

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter the path to the image you want to analyze: ")
    prompt = "Tell me about this image."
    print(f"Analyzing {image_path} with LLaVA...")
    result = llava_image_understanding(image_path, prompt)
    print("\nLLaVA result:")
    print(result) 