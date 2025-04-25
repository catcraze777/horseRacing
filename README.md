# horseRacing
A simple python program based on the social media trend. "Horses" start in an arena loaded from an image and proceed to move and bounce around. Mainly created for practice and to demonstrate various image processing and linear algebra techniques in collision detection.

# Usage
To use the program, first install the required python packages in "requirements.txt" via "python -m pip install -r requirements.txt"

To setup the arena, create an RGBA image of any resolution. The transparency (alpha channel) of the image is used for collision detection and RGB values stored in the image are displayed without transparency. Fully transparent pixels (alpha = 0) are considered free space and do not disrupt any horses and their movement. Opaque or translucent pixels (alpha > 0) are considered walls and trigger collisions. Default filename for the arena is "arena.png" but can be changed within the script. Many image editting programs will overwrite the color data in completely transparent pixels for compression so if weird visual artifacts are visible in an arena ensure you're exporting the image with the pixel color data intact.

To setup the horses, create RGBA images of any resolution and name them in a numerical order. Like the arena, the alpha channel is used for calculating collisions. Unlike the arena, horse sprites are displayed with transparency as you might expect. Horse image files are searched for based on a prefix and suffix and increment a value upwards starting from zero. By default, the code looks for files named "piece_n.png" where n is an integer starting at 0 and is incremented by 1 for each horse image file found. The prefix and suffix can be modified in the script like the arena image file.

"starting_positions.txt" specifies the starting coordinates for each horse. Each horse sprite has their origin in the top left corner of the image so if a horse's starting position is specified as "100, 100", the top left pixel of the horse sprite will be at pixel (100, 100). The file format expects one line for each horse with each line formatted as "int, int".

Many parameters of the script can be easily modifed as global constants and the code is organized and commented to aid developers wanting to modify this baseline framework. Helper functions demonstrate how images loaded as numpy arrays are indexed and processed using various numpy and opencv functions. Modification of this basic framework is heavily encouraged for learning purposes but this framework is currently not designed for serious application.

Once configured, running "horseRacing.py" will immediately boot up a screen with the arena and horses in their initial positions. Horses will begin moving immediately based on an initial velocity vector specified in the global constants of "horseRacing.py".
