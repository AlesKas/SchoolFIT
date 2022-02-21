/// Využijeme knihovnu, která je předmětem zadání projektu
import { Tree } from "./tree.mjs";

/// Testovací data, samozřejmě jde jen o příklad a pro finální
/// hodnocení lze očekávat využití jiných dat
let input = [5,7,2131345646,9,4,13,12415486];

/// Knihovna musí poskytovat konstruktor Tree()
/// Tento konstruktor očekává jediný parametr - řadící fukci, která
/// vyhodnocuje, do kterého podstromu přidat novou hodnotu
let t = new Tree((a,b) => a < b);
/// Objekty vytvořené konstruktorem Tree() musí implementovat metodu
/// insertValue(). Ta očekává jediný parametr - nově přidávanou hodnotu.
input.forEach(i => t.insertValue(i));

/// Ukázka použití knihovny:
/// Iterátorů  vrácených generátory preorder(), inorder() a postorder() je
/// možné vytvořit více, vzájemně budou na sobě nezávislé. Předpokládejte,
/// že v průběhu iterace nebudou vkládány nové prvky.

console.log("Mix 2 iterátorů")
let pre1 = t.preorder()
console.log(pre1.next().value);
console.log(pre1.next().value);
let pre2 = t.preorder()
console.log(pre2.next().value);
console.log(pre1.next().value);
console.log(pre2.next().value);
console.log(pre2.next().value);
console.log(pre1.next().value);

/// Ukázka využití iterátoru získaného z generátoru preorder()
console.log("preorder")
for (let n of t.preorder()) {
	console.log(n);
}

/// Ukázka využití iterátoru získaného z generátoru inorder()
console.log("inorder")
for (let n of t.inorder()) {
	console.log(n);
}
/// Ukázka využití iterátoru získaného z generátoru postorder()
console.log("postorder")
for (let n of t.postorder()) {
	console.log(n);
}
