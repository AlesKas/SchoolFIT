/** @module Tree */

/**
* Pomocný prototyp Node
* @class
* @constructor
* @param {T} value Hodnota uložena v uzlu
*/
var Node = function(value) {
    var instance = Object.create(Node.prototype);
    /**
    * Hodnota uzlu.
    * @name Node#value
    * @type {T}
    */
    instance.value = value;

    /**
    * Levý potomek uzlu.
    * @name Node#left
    * @type {Node}
    */
    instance.left = null;

    /**
    * Pravý potomek uzlu.
    * @name Node#left
    * @type {Node}
    */
    instance.right = null;
    return instance;
}

/**
* Konstruktor třídy strom.
* @class
* @constructor
* @param {function (T, T): boolean} fun Funkce použitá k seřazení prvků ve stromu.
*/
export var Tree = function(fun) {

    /**
    * Funkce použitá k seřazení prvků ve stromu.
    * @name Tree#sort_function
    * @type {function (T, T): boolean}
    */
    this.sort_function = fun;

    /**
    * Uzel stromu.
    * @name Tree#root
    * @type {Node}
    */
    this.root = null;
}

/**
* Přidání prvku do stromu.
* @name Tree#insertValue
* @param {T} value - Prvek přidávaný do stromu.
*/
Tree.prototype.insertValue = function(value) {
    var bst = this;
    var node = Node(value);
    if (bst.root == null) {
        bst.root = node;
        return;
    }

    function recursive_insert(tree) {
        if (!bst.sort_function(tree.value, node.value)) {
            if (tree.left == null)
                tree.left = node;
            else
                recursive_insert(tree.left);
        } else if (bst.sort_function(tree.value, node.value)) {
            if (tree.right == null)
                tree.right = node;
            else
                recursive_insert(tree.right);
        }
    }
    recursive_insert(this.root);
}

/**
* Preorder procházení stromu.
*
* @generator
* @function Tree#preorder
* @yields {T} Prvek stromu.
*/
Tree.prototype.preorder = function*() {
    var bst = this;

    function* iterate(node) {
        if (node) {
            yield node.value;
            yield* iterate(node.left);
            yield* iterate(node.right);
        }
    }
    yield* iterate(bst.root);
}

/**
* Inorder procházení stromu.
*
* @generator
* @function Tree#inorder
* @yields {T} Prvek stromu.
*/
Tree.prototype.inorder = function*() {
    var bst = this;

    function* iterate(node) {
        if (node) {
            yield* iterate(node.left);
            yield node.value;
            yield* iterate(node.right);
        }
    }
    yield* iterate(bst.root);
}

/**
* Postorder procházení stromu.
*
* @generator
* @function Tree#postorder
* @yields {T} Prvek stromu.
*/
Tree.prototype.postorder = function*() {
    var bst = this;

    function* iterate(node) {
        if (node) {
            yield* iterate(node.left);
            yield* iterate(node.right);
            yield node.value;
        }
    }
    yield* iterate(bst.root);
}