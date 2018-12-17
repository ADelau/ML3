from numpy import nan
from textblob import TextBlob
from textblob.en.sentiments import PatternAnalyzer

def stringAnalyzer(string):
    """
    Analyze a string with the TextBlob PatternAnalyzer

    Parameters
    ----------
    string: the string to analyze

    Return
    ------
    polarity, subjectivity, the result of the PatternAnalyzer
    """

    polarity = []
    subjectivity = []

    analyzer = PatternAnalyzer()

    for sentence in string:
        analysis = analyzer.analyze(sentence)
        polarity.append(analysis[0])
        subjectivity.append(analysis[1])

    return polarity, subjectivity

def dateConverter(date):
    """
    Convert a date into int

    Parameters
    ----------
    date: the date

    Return
    ------
    number of fay between January the first 1900 and the date
    """

    if date is nan:
        return nan
    if date == 0:
        return 0

    l = date.split('-')
    m = monthToNum(l[1])

    t = int(l[0]) + 30 * m + 365 * (int(l[2]) - 1900)
    return t

def monthToNum(shortMonth):
    """
    Convert a month name into the month number

    Parameters
    ----------
    shortMonth: the month name

    Return
    ------
    the month number
    """
    return{
            'Jan' : 1,
            'Feb' : 2,
            'Mar' : 3,
            'Apr' : 4,
            'May' : 5,
            'Jun' : 6,
            'Jul' : 7,
            'Aug' : 8,
            'Sep' : 9, 
            'Oct' : 10,
            'Nov' : 11,
            'Dec' : 12
    }[shortMonth]

def genderConverter(gender):
    """
    Convert a Gender (F or M) into a int (0 or 1)

    Parameters
    ----------
    gender: the gender string

    Return
    ------
    the gender int
    """
    return{
            'M' : 1,
            'F' : 2,
    }[gender]

def occupationConverter(occupation):
    """
    Convert an occupation into a int id

    Parameters
    ----------
    occupation: the occupation string

    Return
    ------
    the occupation id, -1 if the string does not correspond to an occupation
    """
    return{
            'artist' : 1,
            'administrator' : 2,
            'educator' : 3,
            'student' : 4,
            'engineer' : 5,
            'salesman' : 6,
            'executive' : 7,
            'retired' : 8,
            'writer' : 9, 
            'technician' : 10,
            'doctor' : 11,
            'other' : 12,
            'librarian' : 13,
            'scientist' : 14,
            'healthcare' : 15, 
            'programmer' : 16,
            'entertainment' : 17,
            'homemaker' : 18,
            'mother' : 19, 
            'lawyer' : 20,
            'marketing' : 21,
            'none' : 22,
    }.get(occupation, -1)
