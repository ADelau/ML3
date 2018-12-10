from numpy import nan
from textblob.en.sentiments import PatternAnalyzer

def stringAnalyzer(string):
    polarity = []
    subjectivity = []

    analyzer = PatternAnalyzer()

    for sentence in string:
        analysis = analyzer.analyze(sentence)
        polarity.append(analysis[0])
        subjectivity.append(analysis[1])

    return polarity, subjectivity

def dateConverter(date):
    if date is nan:
        return nan
    if date == 0:
        return 0

    l = date.split('-')
    m = monthToNum(l[1])
    # t = int(l[2])* 3000
    # t = (t + m) * 3000
    # t = t + int(l[0])

    # d = datetime.date(int(l[2]), m, int(l[0]))
    # ref = datetime.date(1900, 1,)
    # t = time.mktime(d.timetuple())

    t = int(l[0]) + 30 * m + 365 * (int(l[2]) - 1900)
    return t

def monthToNum(shortMonth):
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
    return{
            'M' : 1,
            'F' : 2,
    }[gender]

def occupationConverter(occupation):
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
