## How to use

1. Obtain this repository through one of the following methods:
    1. **Clone (recommended)**
        1. [Make sure you have git](https://github.com/elevengs/pipeline/wiki/Git).
        2. In the terminal, [change directories](https://github.com/elevengs/pipeline/wiki/Terminal#changing-directories) to where you want the pipeline to be, and run `git clone https://github.com/elevengs/segmentation`
        3. This will allow you to pull the latest changes by running the simple command `git pull` in the folder instead of re-downloading the code.
    2. Download (not recommended)
        1. Click [this link](https://github.com/elevengs/segmentation/archive/refs/heads/master.zip) and extract the ZIP archive.
2. Create a folder called `checkpoints` inside the `microsam` folder, and place any model checkpoints you have access to there.
3. Install the Python dependencies using [uv](https://docs.astral.sh/uv/):
        1. Make sure [uv is installed](https://github.com/elevengs/pipeline/wiki/uv).
        2. In the terminal, [change directories](https://github.com/elevengs/pipeline/wiki/Terminal#changing-directories) to the `microsam` folder.
        3. Run these commands in order:
            1. `uv venv .venv`
            2. `uv sync`
            3. `source .venv/bin/activate`
4. Start the Napari viewer:
    1. Ensure you have changed directories to `microsam`.
    2. In the terminal, use uv to run `label.py`.
       1. You will need to provide a model checkpoint and an image, like so:
           2. `uv run label.py ./checkpoints/example.pt ../../data/A/B/C/1.nd2`
       2. There are many more options! You can view all of them by running `uv run label.py --help`.  
5. Napari should open.
    1. Usually, at first, you won't see much in the viewer; this is because the model is generating ["embeddings"](https://en.wikipedia.org/wiki/Embedding_(machine_learning)) for the images.
        1. You can see its progress by clicking on the up arrow in the bottom-right hand corner of the screen. 
    2. Once it is done generating embeddings, more layers will appear on the left side of your screen.
    3. Adjust the contrast limits of the viewer if needed by clicking on a layer and using the handles that appear below the list of tools.
    4. If you want to see the model's segmentation, click "Automatic Segmentation" in the panel that opens on the right.
        2. You can also see its progress in doing this.
    5. See [microSAM's 2D Annotation tutorial](https://www.youtube.com/watch?v=9xjJBg_Bfuc) for more help with Napari.
6. Once you are done, either:
    1. Close the window to save your labels automatically: this will produce a file next to your image with a name that ends in `_labels_autosave_XXXXXXXX.png`, where `XXXXXXXX` is the timestamp.
         1. This is so that you don't accidentally overwrite old labels.
         2. The file contains the result of merging labels from `committed_objects` and `auto_segmentation`.
    2. Force-quit Napari: this will not save anything.
