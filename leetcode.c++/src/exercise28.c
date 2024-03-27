/****************************************************************************************
>FileName:      exercise28.c
>Description:   
>Authot:        zhangxu
>Date:          2019.04.04
>Version:       V0.1
****************************************************************************************/

#include <leetcode.h>

using namespace std;

/*****************************************************************
> 暴力法
******************************************************************/
int Solution28::strStr(string haystack, string needle) 
{   
    int m = haystack.size();
    int n = needle.size();

    if(0 == m || 0 == n)
    {
        return -1;
    }
        
    for(int i = 0; i < m-n; i++ )
    {
        int j = 0;
            
        for(j = 0; j < n ; j++)
        {
            if(haystack[i+j] != needle[j])
            {
                break;
            }
        }
            
        if(j == n)
        {
            return i;
        }
    }    
    
    return -1;   
}

/******************************************************************
>KMP方法：原理详细讲解见网址：
>https://blog.csdn.net/yangwangnndd/article/details/89042454
>https://www.cnblogs.com/en-heng/p/5091365.html
******************************************************************/
static int *getNext(string needle)
{
    int i ,j;
    int strSize = needle.size();

    int *f = new int[strSize];
    f[0] = -1;

    for( j = 1; j < strSize; j++)
    {
        for(i = f[j-1]; ; i = f[i])
        {
            if(needle[j] == needle[i+1])
            {
                f[j] = i+1;
                break;
            }
            else if(-1 == i)
            {
                f[j] = -1;
                break;
            }
        }
    }

    return f;
}

int Solution28::strStr_KMP(string haystack, string needle)
{
    int i, j;
    int *f = getNext(needle);

    for(i = 0, j = 0; i < haystack.size() && j < needle.size();)
    {
        if(haystack[i] == needle[j])
        {
            i++;
            j++;
        }
        else if(0 == j)
        {
            i++;
        }
        else
        {
            j = f[j-1] + 1;
        }
    }

    delete f;
    return j == needle.size() ? i - needle.size() : -1;
}