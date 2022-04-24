/** FLP 2020
Toto je ukazkovy soubor zpracovani vstupu v prologu.
Tento soubor muzete v projektu libovolne pouzit.

autor: Martin Hyrs, ihyrs@fit.vutbr.cz
preklad: swipl -q -g start -o flp19-log -c input2.pl
*/


/** cte radky ze standardniho vstupu, konci na LF nebo EOF */
read_line(L,C) :-
	get_char(C),
	(isEOFEOL(C), L = [], !;
		read_line(LL,_),% atom_codes(C,[Cd]),
		[C|LL] = L).


/** testuje znak na EOF nebo LF */
isEOFEOL(C) :-
	C == end_of_file;
	(char_code(C,Code), Code==10).


read_lines(Ls) :-
	read_line(L,C),
	( C == end_of_file, Ls = [] ;
	  read_lines(LLs), Ls = [L|LLs]
	).


/** rozdeli radek na podseznamy */
split_line([],[[]]) :- !.
split_line([' '|T], [[]|S1]) :- !, split_line(T,S1).
split_line([32|T], [[]|S1]) :- !, split_line(T,S1).    % aby to fungovalo i s retezcem na miste seznamu
split_line([H|T], [[H|G]|S1]) :- split_line(T,[G|S1]). % G je prvni seznam ze seznamu seznamu G|S1


/** vstupem je seznam radku (kazdy radek je seznam znaku) */
split_lines([],[]).
split_lines([L|Ls],[H|T]) :- split_lines(Ls,T), split_line(L,H).

/* Převedení do vnitřní reprezentace */
/* Bylo nutné změnit číslování u zadní strany, pro lepší představivost u rotací */
constructCube([[[UP1, UP2, UP3]],
        [[UP4, UP5, UP6]],
        [[UP7, UP8, UP9]],
        [[FRONT1, FRONT2, FRONT3], [RIGHT1, RIGHT2, RIGHT3], [BACK1, BACK2, BACK3], [LEFT1, LEFT2, LEFT3]],
        [[FRONT4, FRONT5, FRONT6], [RIGHT4, RIGHT5, RIGHT6], [BACK4, BACK5, BACK6], [LEFT4, LEFT5, LEFT6]],
        [[FRONT7, FRONT8, FRONT9], [RIGHT7, RIGHT8, RIGHT9], [BACK7, BACK8, BACK9], [LEFT7, LEFT8, LEFT9]],
        [[DOWN1, DOWN2, DOWN3]],
        [[DOWN4, DOWN5, DOWN6]],
        [[DOWN7, DOWN8, DOWN9]]], 
    (
        UP1, UP2, UP3, UP4, UP5, UP6, UP7, UP8, UP9,
        FRONT1, FRONT2, FRONT3, FRONT4, FRONT5, FRONT6, FRONT7, FRONT8, FRONT9,
        RIGHT1, RIGHT2, RIGHT3, RIGHT4, RIGHT5, RIGHT6, RIGHT7, RIGHT8, RIGHT9,
        BACK9, BACK8, BACK7, BACK6, BACK5, BACK4, BACK3, BACK2, BACK1,
        LEFT1, LEFT2, LEFT3, LEFT4, LEFT5, LEFT6, LEFT7, LEFT8, LEFT9,
        DOWN1, DOWN2, DOWN3, DOWN4, DOWN5, DOWN6, DOWN7, DOWN8, DOWN9
    )
).

/* Výpis kostky */
toString((
        UP1, UP2, UP3, UP4, UP5, UP6, UP7, UP8, UP9,
        FRONT1, FRONT2, FRONT3, FRONT4, FRONT5, FRONT6, FRONT7, FRONT8, FRONT9,
        RIGHT1, RIGHT2, RIGHT3, RIGHT4, RIGHT5, RIGHT6, RIGHT7, RIGHT8, RIGHT9,
        BACK9, BACK8, BACK7, BACK6, BACK5, BACK4, BACK3, BACK2, BACK1,
        LEFT1, LEFT2, LEFT3, LEFT4, LEFT5, LEFT6, LEFT7, LEFT8, LEFT9,
        DOWN1, DOWN2, DOWN3, DOWN4, DOWN5, DOWN6, DOWN7, DOWN8, DOWN9
    )):-
	format('~w~w~w~n', [UP1,UP2,UP3]),
    format('~w~w~w~n', [UP4,UP5,UP6]),
    format('~w~w~w~n', [UP7,UP8,UP9]),
    format('~w~w~w ~w~w~w ~w~w~w ~w~w~w~n', [FRONT1,FRONT2,FRONT3,RIGHT1,RIGHT2,RIGHT3,BACK1,BACK2,BACK3,LEFT1,LEFT2,LEFT3]),
    format('~w~w~w ~w~w~w ~w~w~w ~w~w~w~n', [FRONT4,FRONT5,FRONT6,RIGHT4,RIGHT5,RIGHT6,BACK4,BACK5,BACK6,LEFT4,LEFT5,LEFT6]),
    format('~w~w~w ~w~w~w ~w~w~w ~w~w~w~n', [FRONT7,FRONT8,FRONT9,RIGHT7,RIGHT8,RIGHT9,BACK7,BACK8,BACK9,LEFT7,LEFT8,LEFT9]),
    format('~w~w~w~n', [DOWN1,DOWN2,DOWN3]),
    format('~w~w~w~n', [DOWN4,DOWN5,DOWN6]),
    format('~w~w~w~n~n', [DOWN7,DOWN8,DOWN9]).

solution((
    U,U,U,U,U,U,U,U,U,
    F,F,F,F,F,F,F,F,F,
    R,R,R,R,R,R,R,R,R,
    B,B,B,B,B,B,B,B,B,
    L,L,L,L,L,L,L,L,L,
    D,D,D,D,D,D,D,D,D
)).

/* https://www.fyft.cz/speedcube-clanky/rubikova-kostka-znaceni-tahu/ */
/* Není potřeba brát v úvahu opačné pohyby, jelikož stačí zřetězit více stejných pohybů k dosažení stejného efektu */
/* Rotace vrchní hrany doleva, rotace U */
rotation(
    top_edge_left,
    (
        UP1, UP2, UP3, UP4, UP5, UP6, UP7, UP8, UP9,
        FRONT1, FRONT2, FRONT3, FRONT4, FRONT5, FRONT6, FRONT7, FRONT8, FRONT9,
        RIGHT1, RIGHT2, RIGHT3, RIGHT4, RIGHT5, RIGHT6, RIGHT7, RIGHT8, RIGHT9,
        BACK1, BACK2, BACK3, BACK4, BACK5, BACK6, BACK7, BACK8, BACK9,
        LEFT1, LEFT2, LEFT3, LEFT4, LEFT5, LEFT6, LEFT7, LEFT8, LEFT9,
        DOWN1, DOWN2, DOWN3, DOWN4, DOWN5, DOWN6, DOWN7, DOWN8, DOWN9
    ),
    (
        UP7, UP4, UP1, UP8, UP5, UP2, UP9, UP6, UP3,
        RIGHT1, RIGHT2, RIGHT3, FRONT4, FRONT5, FRONT6, FRONT7, FRONT8, FRONT9,
        BACK7, BACK8, BACK9, RIGHT4, RIGHT5, RIGHT6, RIGHT7, RIGHT8, RIGHT9,
        BACK1, BACK2, BACK3, BACK4, BACK5, BACK6, LEFT1, LEFT2, LEFT3,
        FRONT1, FRONT2, FRONT3, LEFT4, LEFT5, LEFT6, LEFT7, LEFT8, LEFT9,
        DOWN1, DOWN2, DOWN3, DOWN4, DOWN5, DOWN6, DOWN7, DOWN8, DOWN9
    )
).

/* Rotace spodní hrany doleva, rotace D */
rotation(
    bottom_edge_left,
    ( 	
        UP1, UP2, UP3, UP4, UP5, UP6, UP7, UP8, UP9,
        FRONT1, FRONT2, FRONT3, FRONT4, FRONT5, FRONT6, FRONT7, FRONT8, FRONT9,
        RIGHT1, RIGHT2, RIGHT3, RIGHT4, RIGHT5, RIGHT6, RIGHT7, RIGHT8, RIGHT9,
        BACK1, BACK2, BACK3, BACK4, BACK5, BACK6, BACK7, BACK8, BACK9,
        LEFT1, LEFT2, LEFT3, LEFT4, LEFT5, LEFT6, LEFT7, LEFT8, LEFT9,
        DOWN1, DOWN2, DOWN3, DOWN4, DOWN5, DOWN6, DOWN7, DOWN8, DOWN9
    ),
    (
        UP1, UP2, UP3, UP4, UP5, UP6, UP7, UP8, UP9,
        FRONT1, FRONT2, FRONT3, FRONT4, FRONT5, FRONT6, RIGHT7, RIGHT8, RIGHT9, 
        RIGHT1, RIGHT2, RIGHT3, RIGHT4, RIGHT5, RIGHT6, BACK7, BACK8, BACK9,
        BACK1, BACK2, BACK3, BACK4, BACK5, BACK6, LEFT7, LEFT8, LEFT9,
        LEFT1, LEFT2, LEFT3, LEFT4, LEFT5, LEFT6, FRONT7, FRONT8, FRONT9,
        DOWN3, DOWN6, DOWN9, DOWN2, DOWN5, DOWN8, DOWN1, DOWN4, DOWN7
    )
).

/* Rotace levé hrany dolů, rotace L */
rotation(
    left_edge_down,
    (
        UP1, UP2, UP3, UP4, UP5, UP6, UP7, UP8, UP9,
        FRONT1, FRONT2, FRONT3, FRONT4, FRONT5, FRONT6, FRONT7, FRONT8, FRONT9,
        RIGHT1, RIGHT2, RIGHT3, RIGHT4, RIGHT5, RIGHT6, RIGHT7, RIGHT8, RIGHT9,
        BACK1, BACK2, BACK3, BACK4, BACK5, BACK6, BACK7, BACK8, BACK9,
        LEFT1, LEFT2, LEFT3, LEFT4, LEFT5, LEFT6, LEFT7, LEFT8, LEFT9,
        DOWN1, DOWN2, DOWN3, DOWN4, DOWN5, DOWN6, DOWN7, DOWN8, DOWN9
    ),
    (
        BACK1, UP2, UP3, BACK4, UP5, UP6, BACK7, UP8, UP9,
        UP1, FRONT2, FRONT3, UP4, FRONT5, FRONT6, UP7, FRONT8, FRONT9,
        RIGHT1, RIGHT2, RIGHT3, RIGHT4, RIGHT5, RIGHT6, RIGHT7, RIGHT8, RIGHT9,
        DOWN1, BACK2, BACK3, DOWN4, BACK5, BACK6, DOWN7, BACK8, BACK9,
        LEFT7, LEFT4, LEFT1, LEFT8, LEFT5, LEFT2, LEFT9, LEFT6, LEFT3,
        FRONT1, DOWN2, DOWN3, FRONT4, DOWN5, DOWN6, FRONT7, DOWN8, DOWN9
    )
).

/* Rotace pravé hrany nahoru, rotace R */
rotation(
    right_edge_up,
    (
        UP1, UP2, UP3, UP4, UP5, UP6, UP7, UP8, UP9,
        FRONT1, FRONT2, FRONT3, FRONT4, FRONT5, FRONT6, FRONT7, FRONT8, FRONT9,
        RIGHT1, RIGHT2, RIGHT3, RIGHT4, RIGHT5, RIGHT6, RIGHT7, RIGHT8, RIGHT9,
        BACK1, BACK2, BACK3, BACK4, BACK5, BACK6, BACK7, BACK8, BACK9,
        LEFT1, LEFT2, LEFT3, LEFT4, LEFT5, LEFT6, LEFT7, LEFT8, LEFT9,
        DOWN1, DOWN2, DOWN3, DOWN4, DOWN5, DOWN6, DOWN7, DOWN8, DOWN9
    ),
    (           
        UP1, UP2, FRONT3, UP4, UP5, FRONT6, UP7, UP8, FRONT9,
        FRONT1, FRONT2, DOWN3, FRONT4, FRONT5, DOWN6, FRONT7, FRONT8, DOWN9, 
        RIGHT7, RIGHT4, RIGHT1, RIGHT8, RIGHT5, RIGHT2, RIGHT9, RIGHT6, RIGHT3, 
        BACK1, BACK2, UP3, BACK4, BACK5, UP6, BACK7, BACK8, UP9, 
        LEFT1, LEFT2, LEFT3, LEFT4, LEFT5, LEFT6, LEFT7, LEFT8, LEFT9,
        DOWN1, DOWN2, BACK3, DOWN4, DOWN5, BACK6, DOWN7, DOWN8, BACK9 
    )
).

/* Rotace přední strany po směru, rotace F */
rotation(
    front_side_clockwise,
    (
        UP1, UP2, UP3, UP4, UP5, UP6, UP7, UP8, UP9,
        FRONT1, FRONT2, FRONT3, FRONT4, FRONT5, FRONT6, FRONT7, FRONT8, FRONT9,
        RIGHT1, RIGHT2, RIGHT3, RIGHT4, RIGHT5, RIGHT6, RIGHT7, RIGHT8, RIGHT9,
        BACK1, BACK2, BACK3, BACK4, BACK5, BACK6, BACK7, BACK8, BACK9,
        LEFT1, LEFT2, LEFT3, LEFT4, LEFT5, LEFT6, LEFT7, LEFT8, LEFT9,
        DOWN1, DOWN2, DOWN3, DOWN4, DOWN5, DOWN6, DOWN7, DOWN8, DOWN9
    ),
    (           
        UP1, UP2, UP3, UP4, UP5, UP6, LEFT9, LEFT6, LEFT3,
        FRONT7, FRONT4, FRONT1, FRONT8, FRONT5, FRONT2, FRONT9, FRONT6, FRONT3,
        UP7, RIGHT2, RIGHT3, UP8, RIGHT5, RIGHT6, UP9, RIGHT8, RIGHT9,
        BACK1, BACK2, BACK3, BACK4, BACK5, BACK6, BACK7, BACK8, BACK9, 
        LEFT1, LEFT2, DOWN1, LEFT4, LEFT5, DOWN2, LEFT7, LEFT8, DOWN3,
        RIGHT7, RIGHT4, RIGHT1, DOWN4, DOWN5, DOWN6, DOWN7, DOWN8, DOWN9 
    )
).

/* Rotace zadní strany po směru, rotace B */
rotation(
    back_side_clockwise,
    (
        UP1, UP2, UP3, UP4, UP5, UP6, UP7, UP8, UP9,
        FRONT1, FRONT2, FRONT3, FRONT4, FRONT5, FRONT6, FRONT7, FRONT8, FRONT9,
        RIGHT1, RIGHT2, RIGHT3, RIGHT4, RIGHT5, RIGHT6, RIGHT7, RIGHT8, RIGHT9,
        BACK1, BACK2, BACK3, BACK4, BACK5, BACK6, BACK7, BACK8, BACK9,
        LEFT1, LEFT2, LEFT3, LEFT4, LEFT5, LEFT6, LEFT7, LEFT8, LEFT9,
        DOWN1, DOWN2, DOWN3, DOWN4, DOWN5, DOWN6, DOWN7, DOWN8, DOWN9
    ),
    (   
        LEFT1, LEFT4, LEFT7, UP4, UP5, UP6, UP7, UP8, UP9,
        FRONT1, FRONT2, FRONT3, FRONT4, FRONT5, FRONT6, FRONT7, FRONT8, FRONT9,
        RIGHT1, RIGHT2, UP1, RIGHT4, RIGHT5, UP2, RIGHT7, RIGHT8, UP3,
        BACK3, BACK6, BACK9, BACK2, BACK5, BACK8, BACK1, BACK4, BACK7,
        DOWN7, LEFT2, LEFT3, DOWN8, LEFT5, LEFT6, DOWN9, LEFT8, LEFT9,
        DOWN1, DOWN2, DOWN3, DOWN4, DOWN5, DOWN6, RIGHT9, RIGHT6, RIGHT3
    )
).

/* Rotace a uložení výsledku */
make_rotation([], Cube, Cube).
make_rotation([CurrentMove | Rotations], Cube, NewState) :-
    make_rotation(Rotations, CurrentState, NewState),
    rotation(CurrentMove, Cube, CurrentState).

/* Vyřešení kostky a uložení požadovaných rotací */
solveCube(Solution, Cube) :-
    make_rotation(Solution, Cube, NewCute),
    solution(NewCute).

/* Výpis každého kroku z řešení */
writeSolution([], _).
writeSolution([CurrentMove | Rotations], Cube) :-
    rotation(CurrentMove, Cube, NewCube), 
    format('Provedena rotace ~w~n', [CurrentMove]),
    toString(NewCube), writeSolution(Rotations, NewCube).

main :-
		read_lines(Lines),
		split_lines(Lines, Lists),
		constructCube(Lists, Cube),
		toString(Cube),
		solveCube(Solution, Cube),
        writeSolution(Solution, Cube),
		halt.