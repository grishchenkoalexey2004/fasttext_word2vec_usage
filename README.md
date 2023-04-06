# Практическое задание номер 4

## Постановка задачи

1. Использовать предобученную модель word2vec на основе fasttext, предрассчитанную на большом интернет-корпусе Common Crowl, для подсчета близостей слов в датасетах wordsim similarity и wordsim-relatedness.

2. Посчитать корреляцию  с человеческими оценками из датасетов с помошью корреляции Спирмента, которая сравнивает между собой порядок расположения элементов в двух списках. Если корреляция =1, то порядок элементов в списках идентичен.
 

## Алгоритм
1. Достать вектора для слов в паре 
2. Вычислить косинус угла между векторами этих слов
3. Создать упорядченный список пар слов по fasttext 
4. С помощью скрипта для подсчета корреляции Спирмена и посчитать корреляцию между списками (человеческим и от fasttext), сделать выводы

## Необходимые ресурсы
1. Ссылка на фаил с векторами - https://fasttext.cc/docs/en/english-vectors.html (данная программа работает с crawl-300d-2M-subword.vec)
2. Ссылка на wordsim датасеты - http://alfonseca.org/eng/research/wordsim353.html
3. numpy
4. scipy

## Вывод программы
1. Значения корреляции
2. Пары слов со значениями их косинусной близости, записанные в фаил results_relatedness.txt и results_similarity.txt
(в results_relatedness.txt записываются пары из wordsim_relatedness_goldstandart.txt в results_similariy ...)


