/****************************************************************************************
>FileName:      test.c
>Description:   
>Authot:        zhangxu
>Date:          2019.04.04
>Version:       V0.1
****************************************************************************************/
#include <iostream>

#include"leetcode.h"

using namespace std;

int main()
{
    Solution28 test;

    cout<<"*******************************************************"<<endl;
    cout<<test.strStr("hello", "ll")<<endl;
    cout<<test.strStr_KMP("hello", "ll")<<endl;

    return 0;
}
