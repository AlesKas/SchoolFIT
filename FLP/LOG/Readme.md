## Doba výpočtu
Doba výpořtu byla testována pomocí Linuxové utility time.
Testování doby výpočtu bylo prováděno na serveru Merlin.
Pro vstupní soubor test1.in, obsahující referenční příklad, trval výpočet přibližně 0,03s.
Pro vstupní soubor test2.in, obsahující kostku, kterou lze složit jedinou rotací, trval výpočet 0,02s.
Tento čas by však šel snížit implementací inverzních rotací.
Nicméně pro soubor test3.in, který obsahuje také kostku, kterou lze dokončit jedinou rotací, trvá doba výpočtu přibližně stejně.
Pro soubor test4.in, který je možné složit dvěmi rotacemi, už však čas trvání narůstá na 10,857s.
Je tedy patrné, že s rostoucím počtem rotací nutných ke složení kostky, roste čas velice rychle.
