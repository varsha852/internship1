#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This python program finds the factorial of a number


# In[2]:


def factorial(num):
    if num == 1:
        return num
    else:
        return num * factorial(num - 1)
    


# In[4]:


# We will find the factorial of this number 
num = int(input("enter a number: "))
if num < 0:
    print("factorial cannot be found for negative number ")
elif num == 0:
    print("factorial of 0 is 1")
else:
    print("factorial of" , num, "is: ", factorial(num))


#  Write a python program to find whether a number is prime or composite.
# 

# In[7]:


#Input a number and check if the number is prime or composite number
n= int(input("Enter any number:"))
if(n ==0 or n == 1):
    printf(n,"Number is neither prime nor composite")
elif n>1 :
    for i in range(2,n):
        if(n%i == 0):
            print(n,"is not prime but composite number")
            break
    else:
        print(n,"number is prime but not composite number")
else :
    print("Please enter positive number only ")


# Write a python program to check whether a given string is palindrome or not.

# In[8]:


st = input("you are beutiful the way you are : ")

if(st == st[:: - 1]):
   print("This is a Palindrome String")
else:
   print("This is Not")


# Write a Python program to get the third side of right-angled triangle from two given sides.
# 

# In[1]:


def pythagoras(opposite_side,adjacent_side,hypotenuse):
        if opposite_side == str("x"):
            return ("Opposite = " + str(((hypotenuse**2) - (adjacent_side**2))**0.5))
        elif adjacent_side == str("x"):
            return ("Adjacent = " + str(((hypotenuse**2) - (opposite_side**2))**0.5))
        elif hypotenuse == str("x"):
            return ("Hypotenuse = " + str(((opposite_side**2) + (adjacent_side**2))**0.5))
        else:
            return "You know the answer!"
    
print(pythagoras(3,4,'x'))
print(pythagoras(3,'x',5))
print(pythagoras('x',4,5))
print(pythagoras(3,4,5))


# Write a python program to print the frequency of each of the characters present in a given string

# In[9]:


string=input("enter the string ")
freq=[None]*len(string)

for i in range(0,len(string)):
    freq[i]=1
    for j in range(i+1,len(string)):
        if(string[i]==string[j]):
            freq[i]=freq[i]+1
            
            string=string[:j]+'0'+string[j+1:];

print("character and their frequency");
for i in range(0,len(freq)):
    if(string[i]!=''and string[i]!='0'):
        print(string[i]+"="+str(freq[i]))


# In[ ]:




