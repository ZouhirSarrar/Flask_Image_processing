# ImSight
## Python image search engine 
This a web application, using Flask as the server, that allows you to
to implement the basic functionality of an indexing system, and to
search for images by content using colour-based descriptors
and texture.
The development of an image indexing and retrieval system by the
content consists of four phases:

- Create our descriptors(Histogram calculation, Haralick filter, gabor filter)

- The indexing phase, for the calculation of characteristics from the our dataset of images (famous places dataset)
[Dataset link](https://drive.google.com/uc?id=1Z7kDsK_7ko_1mMyAid92kQt-tlHQjh8Q&export=download)

- Defining similarity using euclidean distance to compare our vectors that we already extracted.

- The search phase, which consists in extracting the vector of descriptors of
the requested image and compare it with the vectors of the calculated descriptors.

