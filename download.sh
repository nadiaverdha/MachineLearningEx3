# download data files
mkdir news_data
curl -o news_data/20news-bydate.tar.gz http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
gunzip -c news_data/20news-bydate.tar.gz | tar xopf - -C news_data


mkdir reuters_data
curl -o reuters_data/euters21578.tar.gz http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz
gunzip -c reuters_data/euters21578.tar.gz | tar xopf - -C reuters_data