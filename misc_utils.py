# -*- coding: utf-8 -*-
#
# (c) Koninklijke Philips Electronics N.V. 2016
#
# All rights are reserved. Reproduction or transmission in whole or in part,
# in any form or by any means, electronic, mechanical or otherwise, is
# prohibited without the prior written permission of the copyright owner.
#
#    Filename: misc_utils.py
#
# Description: A collection of miscellaneous utility functions
#
#      Author: Alexander Fischer, 2016-04-15
#

from __future__ import print_function

import csv
import datetime
import difflib
import json
import os
import re
import subprocess
import time

timeStampMode = "Clock"
logFile = None

def TimeStamp():
    """
    Time stamp relative or absolute for primarily for logging.
    """

    global timeStampMode
    if timeStampMode == "Clock":
        timeStamp = "%08.3f" % time.clock()
    else:
        timeStamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return timeStamp

def Log(level, msg):
    """
    Log message.
    """
    
    global logFile
    logLine = TimeStamp() + ' ' + level + ': ' + msg
    if logFile == None:
        print(logLine)
    else:
        logFile.write(logLine + '\n')

def LogInfo(msg):
    """
    Information log message with time stamp.
    """
    
    Log("INFO", msg)

def LogWarn(msg):
    """
    Warning log message with time stamp.
    """
    
    Log("WARNING", msg)

def LogError(msg):
    """
    Error log message with time stamp.
    """
    
    Log("ERROR", msg)

def SetLogFile(fileName, append = False):
    """
    Set the log file for the Log.... methods.
    """
    
    global logFile
    if logFile != None:
        logFile.close()
        logFile = None
    if fileName != None and fileName != '':
        if append and os.path.exists(fileName):
            logFile = open(fileName, 'at')
        else:
            logFile = open(fileName, 'wt')
        if logFile == None:
            LogError('cannot open log file ' + fileName)
    
def Print(obj, indent = '', prefix = ''):
    """
    Print a list/dictionary hierarchy in a more readable fashion.
    """
    
    if type(obj) == type({}):
        print(indent + prefix + '{')
        for key in obj.keys():
            Print(obj[key], indent + '  ', '"' + str(key) + '" : ')
        print(indent + '}')
    elif type(obj) == type([]) or type(obj) == type(set()):
        print(indent + prefix + '[')
        for item in obj:
            Print(item, indent + '  ')
        print(indent + ']')
    elif type(obj) == type(()):
        print(indent + prefix + '(')
        for item in obj:
            Print(item, indent + '  ')
        print(indent + ')')
    else:
        now = datetime.datetime.now()
        value = obj
        if type(obj) == type(''):
            value = '"' + value + '"'
        elif type(obj) == type(u''):
            value = 'u"' + value + '"'
        elif type(obj) == type(now):
            # strftime cannot handle years before 1900
            if obj < datetime.datetime(1900, 1, 1):
                value = obj.replace(year=1900).strftime('%Y-%m-%d %H:%M:%S')
            else:
                value = obj.strftime('%Y-%m-%d %H:%M:%S')
        try:
            print(indent + prefix + str(value))
        except:
            print(indent + prefix, value)

def PrintBig(msg, font='banner', width=120):
    """
    Print a message with BIG letters.
    Requires termcolor and pyfiglet module.
    """
    
    import termcolor
    import pyfiglet
    termcolor.cprint(pyfiglet.figlet_format(msg, width=width, font=font))

def ExtractDateTime(fileName):
    """
    Extract datetime from a fileName or string in various formats.
    """
    
    b = os.path.basename(fileName)
    dateTime = ''
    dateTimeMatch = re.search('[12]\d\d\d[01]\d[0-3]\d_[012]\d[0-5]\d', b)
    if dateTimeMatch:
        dateTime = dateTimeMatch.group(0)
    if dateTime == '':
        dateTimeMatch = re.search('[12]\d\d\d-[01]\d-[0-3]\d_[012]\d[0-5]\d', b)
        if dateTimeMatch:
            dateTime = dateTimeMatch.group(0)
    if dateTime == '':
        dateTimeMatch = re.search('[12]\d\d\d[01]\d[0-3]\d', b)
        if dateTimeMatch:
            dateTime = dateTimeMatch.group(0) + '_0000'
    if dateTime == '':
        dateTimeMatch = re.search('[12]\d\d\d-[01]\d-[0-3]\d', b)
        if dateTimeMatch:
            dateTime = dateTimeMatch.group(0) + '_0000'
    dt = None
    if dateTime != '':
        dateTime = dateTime.replace('-', '')
        try:
            dt = datetime.datetime.strptime(dateTime, '%Y%m%d_%H%M')
        except:
            dt = None
    return dt

def GetFileList(dir):
    """
    Get the list of files in a directory (including subdirs).
    """
    
    fileList = []
    for root, dirs, files in os.walk(dir):
        if len(files) > 0:
            for f in files:
                fileList.append(os.path.join(root,f))
    return fileList

def GetDirList(dir):
    """
    Get the list of directories in a directory (including subdirs).
    """

    dirList = []
    for root, dirs, files in os.walk(dir):
        if len(dirs) > 0:
            for d in dirs:
                dirList.append(os.path.join(root,d))
    return dirList
    
def PrintTable(table, header = True, format = None):
    """
    Print a well formatted table.
    """

    global logFile
    defaultFormat = '20l' # width 20, left justified
    if format == None:
        format = []
        for key in table[0].keys():
            format.append(key + ':' + defaultFormat) 
    totalWidth = 0
    for f in format:
        tmp = f.split(':')
        name = tmp[0]
        if len(tmp) > 1:
            w = tmp[1]
        else:
            w = defaultFormat
        totalWidth = totalWidth + int(w[:-1]) + 1
    if header:
        line = ''
        for key in format:
            tmp = key.split(':')
            name = tmp[0]
            if len(tmp) > 1:
                w = tmp[1]
            else:
                w = defaultFormat
            if int(w[:-1]) < len(name):
                name = eval('name[:' + str(w[:-1]) + ']')
            if w[-1] == 'c':
                just = 'center(' + str(w[:-1]) + ')'
            else:
                just = w[-1] + 'just(' + str(w[:-1]) + ')'
            if line != '':
                line = line + ' '
            line = line + eval('name.' + just)
        print(line)
        print('='.ljust(totalWidth, '='))
        if logFile != None:
            logFile.write(line + '\n')
            logFile.write('='.ljust(totalWidth, '=') + '\n')
    for row in table:
        if row == None:
            continue
        line = ''
        for key in format:
            tmp = key.split(':')
            name = tmp[0]
            if len(tmp) > 1:
                w = tmp[1]
            else:
                w = defaultFormat
            valueFormat = ''
            if len(tmp) > 2:
                valueFormat = tmp[2]
            if ':' in w:
                [ w, valueFormat ] = w.split(':')
            if w[-1] == 'c':
                just = 'center(' + str(w[:-1]) + ')'
            else:
                just = w[-1] + 'just(' + str(w[:-1]) + ')'
            if line != '':
                line = line + ' '
            if valueFormat != '':
                line = line + eval('unicode(\'{0:' + valueFormat + '}\'.format(row[name])).' + just)
            else:
                line = line + eval('unicode(row[name]).' + just)
        print(line)
        if logFile != None:
            logFile.write(line + '\n')
    
def ReadCsv(fileName, delim=';', asDict=True):
    """
    Read CSV file.
    """
    
    result = []
    with open(fileName, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delim)
        l = 1
        header = next(reader)
        if not asDict:
            result.append(header)
        for row in reader:
            l += 1
            if len(row) != len(header):
                print('ERROR:', fileName + ':' + str(l), ';'.join(row))
            entry = row
            if asDict:
                entry = {}
                for i in range(len(row)):
                    entry[header[i]] = row[i]
            result.append(entry)
    return result

def ReadLocked(fileName):
    """
    Read a JSON file taking into account a lock indicator.
    """
    
    while os.path.exists(fileName + '.lck'):
        time.sleep(0.05)
    fp = open(fileName, 'rt')
    data = json.load(fp)
    fp.close()
    return data

def WriteLocked(fileName, data):
    """
    Write a JSON file including a lock indicator.
    """
    
    dirName = os.path.dirname(fileName)
    if dirName != '' and not os.path.isdir(dirName):
        os.makedirs(dirName)
    while os.path.exists(fileName + '.lck'):
        time.sleep(0.05)
    fp = open(fileName + '.lck', 'wt')
    fp.write('LOCKED')
    fp.close()
    fp = open(fileName, 'wt')
    json.dump(data, fp, indent=2, sort_keys=True)
    fp.close()
    try:
        os.remove(fileName + '.lck')
    except:
        pass

def WriteCsv(data, fileName, cols = []):
    """
    Write a list of data to a CSV file.
    """

    with open(fileName, 'w', encoding='utf-8', newline='') as f:
        csvwriter = csv.writer(f, delimiter=';')
        if len(data) > 0 and type(data[0]) == type({}):
            # data is a list of dicts -> extract header row from keys
            if len(cols) == 0:
                header = sorted(data[0].keys())
            else:
                header = cols
            csvwriter.writerow(header)
        for d in data:
            row = []
            if type(d) == type({}):
                for col in header:
                    row.append(d[col])
            else:
                row = d
            csvwriter.writerow(row)

def DataType(obj):
    """
    Get a string representing the type of the argument object.
    """
    
    result = ''
    if obj != None:
        result = str(type(obj)).replace('<type \'', '').replace('\'>', '')
    return result
            
def DateTime(dateTime):
    """
    Convert string into datetime.
    """

    result = dateTime
    now = datetime.datetime.now()
    if not type(result) == type(now):
        if 'T' in result:
            result = dateTime.replace('T', ' ')
        try:
            if '.' in result:
                p = result.index('.')
                result = result[:p]
            if len(result) == 10:
                result = datetime.datetime.strptime(result, '%Y-%m-%d')
            else:
                result = datetime.datetime.strptime(result, '%Y-%m-%d %H:%M:%S')
        except:
            result = now.replace(microsecond=0)
    return result
    
def DateTimeToString(data):
    """
    Convert all datetime types in a list of disctionaries to string.
    """
    
    now = datetime.datetime.now()
    for x in data:
        for key in x.keys():
            value = x[key]
            if type(value) == type(now):
                value = value.isoformat().replace('T', ' ')
                if '.' in value:
                    p = value.index('.')
                    value = value[:p]
            x[key] = value

def StringToDateTime(data, key=None):
    """
    Convert all datetime strings in a list of disctionaries to DateTime.
    """
    
    for x in data:
        for k in x.keys():
            value = x[k]
            if (key == None and 'datetime' in k.lower()) or k == key:
                value = DateTime(value)
            x[k] = value

def KeyName(name):
    """
    Convert an arbitrary name (including spaces) to a CamelCase key name (without spaces) starting with a lowercase character.
    e.g. 'Time Of Day' -> 'timeOfDay'
    """

    result = name.title().replace(' ', '')
    result = result[0].lower() + result[1:]
    return result

def FindData(data, match, fuzzy=False):
    """
    Find entries using multiple match criteria in a list of dictionaries.
    """

    result = []
    if not type(match) == type({}):
        print('ERROR: match must be a dictionary')
        return result

    def __match(elem, match, ignoreCase=False, ignoreSpace=False, fuzzy=False):
        result = True
        if fuzzy:
            result = 0.0
        for key in match.keys():
            if not key in elem.keys():
                result = False
                break
            x = elem[key]
            m = match[key]
            if x == None:
                result = False
                break
            if ignoreCase:
                x = x.lower()
                m = m.lower()
            if ignoreSpace:
                x = x.replace(' ', '')
                m = m.replace(' ', '')
            if fuzzy:
                r = difflib.SequenceMatcher(None, x, m).ratio()
                if r > result:
                    result = r                
            elif not x == m:
                result = False
                break            
        return result

    for x in data:
        if __match(x, match):
            result.append(x)

    if len(result) == 0:
        for x in data:
            if __match(x, match, True):
                result.append(x)

    if len(result) == 0:
        for x in data:
            if __match(x, match, True, True):
                result.append(x)

    if len(result) == 0 and fuzzy:
        rMax = 0.0
        res = None
        for x in data:
            r = __match(x, match, True, True, True)
            if r > rMax:
                rMax = r
                res = x
        if res != None:
            result.append(res)
        
    return result

def SortData(data, keys):
    """
    Sort a list of data according to a list of keys.
    """

    from operator import itemgetter
    result = sorted(data, key=eval('itemgetter(\'' + '\',\''.join(keys) + '\')'))
    return result

def TimeFilterData(data, t=None, dur=0):
    """
    Extract a time interval from the list of data values.
    """

    result = []
    if t == None:
        t = datetime.datetime.utcnow()
    delta = datetime.timedelta(seconds=0)
    t2 = t
    if type(dur) == type(delta):
        t2 += dur
    else:
        t2 += datetime.timedelta(seconds=dur)
    t1 = t
    if t2 < t1:
        t1 = t2
        t2 = t
    for x in data:
        if not 'DateTime' in x.keys():
            continue
        dt = DateTime(x['DateTime'])
        if dt < t1 or dt > t2:
            continue
        if not x in result:
            result.append(x)
    return result

def GetIpConfig():
    cmd = [ 'ipconfig', '/all' ]
    ipConfig = subprocess.Popen(cmd, shell=True, bufsize=4096, stdout=subprocess.PIPE).stdout
    result = {}
    adapter = None
    for line in ipConfig.readlines():
        m = re.match('^([^\s][^:]+):', str(line))
        if m:
            adapter = m.group(1)
            result[adapter] = {}
        m = re.match('^\s+([^:]+)\s:\s(.*)', str(line))
        if adapter != None and m:
            key = m.group(1).replace(' .', '').strip().strip('.')
            value = m.group(2).strip()
            result[adapter][key] = value
    return result
    
def GetLocalIp():
    result = ''
    config = GetIpConfig()
    for adapter in config.keys():
        if adapter.startswith('Ethernet') and adapter.endswith('Local Area Connection') and 'IPv4 Address' in config[adapter].keys():
            result = config[adapter]['IPv4 Address'].split('(')[0]
    if result == '':
        for adapter in config.keys():
            if adapter.startswith('Wireless LAN') and adapter.endswith('Wireless Network Connection') and 'IPv4 Address' in config[adapter].keys():
                result = config[adapter]['IPv4 Address'].split('(')[0]
    return result

if __name__ == '__main__':
    testData = { 'abc' : 123, 'def' : 'some string', 'someList' : [ 'a', 'b', 'c' ], 35 : 'bla', 'a mixed tuple' : ( 74, 'sdf', 235 ) }
    Print(testData)
    Print(GetIpConfig())
    print('IP =', GetLocalIp())
