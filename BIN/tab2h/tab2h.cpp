//---------------------------------------------------------------------------
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <string>

using namespace std;

unsigned int vstupu = 0;
unsigned int vystupu = 0;
int    *pole = NULL;
char   buff[16];
int    pocint = 0;
int    typgen=1; //typ generovaneho vystupu (0-seznam integeru / 1-c pole)

//---------------------------------------------------------------------------
void vypis() {
  for (unsigned int i=0; i < vstupu+vystupu; i++) {
      if ((i == vstupu) && (typgen==0))
         printf("\n");

      if ((typgen == 1) && (pocint>0)) { //!tiskne o jeden vypis zpet - aby ze na konci zrusilo lomitko
         printf("  a[%d]=%s;\\\n",pocint-1,buff);
      }

      sprintf(buff,"0x%.8x",pole[i]);
      if (typgen == 0)
         printf("%s,",buff);

      pocint++;
  }
  if (typgen == 0)
     printf("\n\n");
}

int main(int argc, char* argv[])
{
    FILE    *soubor = NULL;
    char    ch;
    int     ch1;
    int     rpoc = 0; //pocet nactenych bitu (odpovida poctu radek s daty % 32)
    int     radek = 1; //pocitadlo radek souboru
    bool    stav_vstup = true;
    int     paridx;
    bool    dpopis = true;
    bool    prevkom = false;
    bool    prevdir = false;
    string   svst, svyst, spopis;

    for (int i=1; i < argc;i++) {
        if (strcmp(argv[i],"-i") == 0)
            typgen=0;
        else {
            soubor = fopen(argv[i],"r");
            break;
        }
    }

    if (soubor == NULL)
        soubor = stdin;

    do {
       ch = getc(soubor);
       if (ch == '#') {
          if ((ch = getc(soubor)) == '%') { //osetrit %
            if (dpopis) {
              if (prevkom) spopis += "\\n";
              spopis += "#%%";
            }
            prevdir = true;
          } else {
            if (dpopis) { if (prevdir) spopis += "\\n"; else spopis += ' '; }
            ungetc(ch,soubor);
            prevdir = false;
          }
          while ((ch != '\n') && ((ch = getc(soubor)) != EOF)) {
            if ((dpopis) && (ch!='#') && (ch!='\r')  && (ch!='\n'))
               spopis += ch;
          }
          prevkom = true;
       } else if (isspace(ch)) {
          dpopis = false;
          while ((ch != '\n') && ((ch = getc(soubor)) != EOF) && (isspace(ch)));
          if (ch != '\n')
             ungetc(ch, soubor);
       } else if (ch == ':') {
          stav_vstup = !stav_vstup;
       } else if (ch != EOF) {
          dpopis = false;
          if ((ch != '0') && (ch != '1')) {
             fprintf(stderr,"r.%d: Chybny znak v datovem souboru!",radek);
             return -1;
          }
          if (stav_vstup) { //cte se vstupni kombinace
             svst += ch;
          } else { //cte se vystupni kombinace
             svyst += ch;
          }
       }

       if ((!stav_vstup) && ((ch == '\n') || (ch == EOF))) {
          stav_vstup = true;
          if ((svst != "") && (svyst != "")) {
             if (rpoc == 0) { //pri nacteni prvni radky se musi alokovat pole
                vstupu = svst.length();
                vystupu = svyst.length();
                if ((vstupu == 0) || (vystupu == 0)) {
                   fprintf(stderr,"r.%d: Chybny pocet vstupu / vystupu!",radek);
                   return -1;
                }
                pole = new int [vstupu + vystupu];
                //for (int i=0; i < vstupu + vystupu; i++) pole[i] = 0;
                if (typgen == 0) {
                   printf("Vstupu: %d Vystupu: %d \n",vstupu, vystupu);
                } else {
                   if ((spopis.length() > 0) && (spopis[0] != '#')) spopis.insert(spopis.begin(),'#');
                   printf("#define POPIS \"%s\\n\"\n",spopis.c_str());
                   printf("//Pocet vstupu a vystupu\n");
                   printf("#define PARAM_IN %d        //pocet vstupu komb. obvodu\n",vstupu);
                   printf("#define PARAM_OUT %d       //pocet vystupu komb. obvodu\n",vystupu);
                   printf("//Inicializace dat. pole\n");
                   printf("#define init_data(a) \\\n");
                }
             }

             if ((vstupu != svst.length()) || (vystupu != svyst.length())) {
                fprintf(stderr,"r.%d: Nesouhlasi pocet vstup/vystupnich hodnot!",radek);
                return -1;
             }

             for (unsigned int i=0; i < vstupu; i++) {
                ch1 = svst[i] - 48;
                pole[i] = ((pole[i] >> 1) & 0x7fffffff) | (ch1*0x80000000);
             }
             for (unsigned int i=0; i < vystupu; i++) {
                ch1 = svyst[i] - 48;
                pole[i+vstupu] = ((pole[i+vstupu] >> 1) & 0x7fffffff) | (ch1*0x80000000);
             }

             if (rpoc % 32 == 31) { //pokud je pole integeru naplneno, musi se vypsat
                vypis();
             }

             rpoc++;
             svst = "";
             svyst = "";
          }
          radek++;
       }
    } while (ch != EOF);

    if ((rpoc >0) && (rpoc % 32 != 0))  {  // je-li pole nezaplneno je treba doplnit do 32 bitu
       int imax = (32 - (rpoc % 32));      // posledni zapsanou bitovou kombinaci
       if (typgen==0) printf("# doplneni %d bitu \n", imax);
       for (int i=0; i < imax; i++)
          for (unsigned int j=0; j < vstupu+vystupu; j++) {
              ch1 = pole[j] & 0x80000000;
              pole[j] = ((pole[j] >> 1) & 0x7fffffff) | (ch1);
          }
       vypis();
    }
    printf("  a[%d]=%s;\n",pocint-1,buff); //vypis posledni nevypsane 

    if (typgen==0)
       printf("#pocet integeru: %d\n\n",pocint);
    else {
       printf("//Pocet prvku pole\n");
       printf("#define DATASIZE %d\n",pocint);
    }
    fclose(soubor);

    if (pole != NULL)
       delete[] pole;

//    getchar();

    return 0;
}
//---------------------------------------------------------------------------
