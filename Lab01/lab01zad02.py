import pandas as pd
import matplotlib.pyplot as plt


miasta = pd.read_csv('miasta.csv')

# print(miasta)
# print(miasta.values)

new_row = pd.Series([2010,460,555,405], index=miasta.columns)
miasta = miasta.append(new_row, ignore_index=True)


#plot miasta using matplotlib
#plt.plot(miasta['Rok'], miasta['Gdansk'], 'r.-')
#plt.xlabel("Lata")
#plt.ylabel("Liczba mieszkańców")
#plt.title("Liczba mieszkańców w Gdańsku")
#plt.show()

axis = miasta.plot(x='Rok', y=miasta.columns[1:], kind='line', title='Liczba mieszkańców w miastach Polski', marker="o")
axis.set_xlabel("Lata")
axis.set_ylabel("Liczba mieszkańców")
plt.show()


miasta = miasta.set_index('Rok')
#standaryzacja danych
standard = (miasta - miasta.mean()) / miasta.std()
print(standard)
#srednia
print("Srednia: ")
print(standard.mean())
#odchylenie standardowe
print("Odchylenie standardowe: ")
print(standard.std())

#normalizacja danych
normal = (miasta - miasta.min()) / (miasta.max() - miasta.min())
print(normal)
#minimum
print("Minimum: ")
print(normal.min())
#maksimum
print("Maksimum: ")
print(normal.max())
