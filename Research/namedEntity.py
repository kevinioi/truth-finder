#not that great, but need to find and remove people/place names for training stance classifier

import nltk
from nltk.tokenize import word_tokenize

sent = "Seventeen-year-old Holden Caulfield lives in an unspecified institution in Southern California, near Hollywood, in 1951. He intends to \
    live with his brother D.B., an author and World War II veteran with whom Holden is angry for becoming a screenwriter, one month after his \
    discharge. Holden recalls the events of the previous Christmas. Holden begins his story at Pencey Preparatory Academy, a boarding school \
    in Agerstown, Pennsylvania. At sixteen in 1950, Holden has been expelled due to poor work and is not allowed to return after Christmas break.\
    He plans to return home on that day so that he will not be present when his parents receive notice of his expulsion. After forfeiting a fencing\
    match in New York by forgetting the equipment on the subway, he is invited to the home of his history teacher, Mr. Spencer, who is a \
    well-meaning but long-winded old man. Spencer greets him and offers him advice, but he embarrasses Holden by criticizing Holden history."

words = word_tokenize(sent)

print(nltk.ne_chunk(nltk.pos_tag(words),binary=True))