# Trabalho Ciência de Dados

## Classificação

Age (numeric)
Sex (text: male, female)
Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
Housing (text: own, rent, or free)
Saving accounts (text - little, moderate, quite rich, rich)
Checking account (numeric, in DM - Deutsch Mark)
Credit amount (numeric, in DM)
Duration (numeric, in month)
Purpose (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)

                 Tipo de Dado           Escala     Cardinalidade Valores
Age              quantitativa/numerica  razao      discreta      -
Sex              qualitativa/categorica nominal    nominal       male, female
Job              qualitativa/categorica ordinal    ordinal       0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled
Housing          qualitativa/categorica nominal    nominal       own, rent, or free
Saving accounts  qualitativa/categorica ordinal    ordinal       little, moderate, quite rich, rich
Checking account qualitativa/categorica ordinal    ordinal       little, moderate, rich
Credit amount    quantitativa/numerica  razao      discreta      -
Duration         quantitativa/numerica  razao*     discreta      -
Purpose          qualitativa/categorica nominal    nominal       car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others

Obs.:
* A duração é um caso especial que apesar de não ter zero absoluto (tempo tem zero relativo), leva à interpretação de ser intervalar, mas podemos encontrar proporcionalidade e nos leva à interpretação de escala de medida racional.
