## BKG-2-CNF

FLP 2021/2022 Letní semestr
xkaspa48 Aleš Kašpárek

### Implementace
Implementace projektu byla rozdělena do několika souborů, kde každý jeden soubor má na starosti různou funkčnost programu.\
Main.hs:\
Hlavní vstupní bod programu, má na starosti načtení argumentů a následné vypsání na standartní výstup podle volby argumentu.\
GrammarData.hs:\
Datové struktury potřebné k uchování gramatiky jazyka. Obsahuje typy Rule, který odpovídá přepisovacímu pravidlu gramatiky, který je složen z levé a pravé části.
Pro zjednodušení implementace je typ pravé strany přepisovacího pravidla [String], což pak napomáhá k jednoduššímu zpracování pravé strany při odebírání jedoduchých pravidel, tak při konečném převodu na CNF.
Dále je pak implementován typ Grammar, které reprezentuje formální gramatiku.\
Parse.hs:\
Soubor sloužící k parsování vstupu a jeho následnému uložení do typu Grammar.\
IsValid.hs:\
Soubor sloužící k ověření, že na vstupu je validní formální gramatika.\
V tomto souboru je kontrolováno, zda-li jsou všechny neterminály, přítomné v množině neterminálích symbolů, jsou maximální délky 1 a jsou to písmena od A do Z. Podobná kontrola je pak provedena s terminálními symboly. Následně jsou pak kontrolována přepisující pravidla, jestli obsahují validní terminální symboly jak na levé, tak na pravé straně, a zda-li jsou pak na pravé straně přítomny validní terminální symboly.\
Simple.hs:\
V tomto souboru je vytvořena nová gramatika, které ze vstupní gramatiky odstraní jednoduchá pravidla. Jednoduchá pravidla jsou tvaru Neterminál->Neterminál. Pro každé pravidlo je zde testováno, zda-li je jednoduché, pokud ano, je pro něj sestavena množina N<sub>a</sub>, které obsahuje neterminální symboly na pravé straně jednoduchých pravidel tohoto neterminálu na levé straně. Následně jsou pak do této nové gramatiky vloženy pravidla, které nejsou jednoduchá a nová pravidla, která vznikly na základě množiny N<sub>a</sub> neterminálního symbolu.\
Cnf.hs:\
Soubor vytvářející novou gramatiku ze vstupní gramatiky, které se nachází v Chomskeho normální formě. Chomskeho normánlí forma obsahuje pouze pravidla ve tvaru Neterminál->Terminál nebo Neterminál->NeterminálNeterminál. Pro vytvoření CNF je nutné napřed vstupní gramatiku zbavit jednoduchých pravidel. Následně jsou pak přímo v tomto souboru vytvořena nová pravidla, které se nachází v CNF formě. Dochází zde k rozdělení pravidel, kde se na pravé straně nachází 3 a více symbolů. Proto jsou vytvářeny nové neterminální symboly ve tvaru &lt;2Symboly&gt;. Toto vytváření nových symbolů je pak následně rekurzivně opakováno, dokud nedojde ke stavu, kdy se na pravé straně nachází pouze dva neterminální symboly. Následně je poté na výsledek aplikována funkce, která vytvoří nové neterminální sympboly pro terminální symboly, které nestojí a pravé straně pravidel samotné. Pro ně jsou vytvořeny nové neterminální symboly ve tvaru &lt;původníSymbol&gt;. Nakonec jsou pak posbírány všechny neterminální symboly stojící na levých stranách pravidel, a vloženy do množiny neterminálních symbolů této nové gramatiky.

### Spuštění
Projekt je nutné přeložit pomocí příkazu make, spuštěného v kořenové složce projektu, který přeloží všechny soubory a vytvoří spustitelný soubor.\
Následné spuštění je pak provedeno příkazem *./flp21-fun volby [vstup]* kde argument volby může nabývat tvaru:\
-i vypíše na standartní výstup reprezentaci načtené vstupní gramatiky\
-1 vypíše na standartní výstup reprezentaci gramatiky, která vznikla ze vstupní gramatiky odstraněním jednoduchých pravidel\
-2 vypíše na standartní výstup reprezentaci gramatiky, která vznikla ze vstupní gramatiky převodem do chomskeho normální formy\
a argument *[vstup]* udává vstupní soubor obsahující vstupní gramatiku. Pokud argument *[vstup]* není specifikován, program načístá vstupní gramatiku ze standartního vstupu.

