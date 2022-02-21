class Node {
    constructor(value) {
        this.value = value;
        this.left = null;
        this.right = null;
    }
}

export class Tree {
    constructor(fun) {
        this.root = null;
        this.sort_function = fun;
    }

    insertValue(value) {
        let node = new Node(value);
        if (this.root === null) {
            this.root = node;
        } else {
            this.#insert(this.root, node);
        }
    }

    #insert(startNode, newNode) {
        if (!this.sort_function(startNode.value, newNode.value)) {
            if (startNode.left === null) {
                startNode.left = newNode;
            } else {
                this.#insert(startNode.left, newNode);
            }
        } else {
            if (startNode.right === null) {
                startNode.right = newNode;
            } else {
                this.#insert(startNode.right, newNode);
            }
        }
    }

    *preorder() {
        const root = this.root;
        const stack = [root];
        while (stack.length > 0) {
            const node = stack.shift();
            yield node.value
            if (node.left) {
                stack.push(node.left)
            }
            if (node.right) {
                stack.push(node.right)
            }
        }
    }

    *inorder() {
        const root = this.root;
        const stack = [];
        let current = root;
        while (stack.length > 0 || current != null) {
            while (current != null) {
                stack.push(current);
                current = current.left;
            }
            current = stack.pop();
            yield current.value;
            current = current.right
        }
    }

    *postorder() {
        const root = this.root;
        const stack = [root];
        const stack2 = []
        while (stack.length > 0) {
            let current = stack.pop();
            stack2.push(current);
            if (current.left != null)
                stack.push(current.left);
            if (current.right != null)
                stack.push(current.right);
        }
        while (stack2.length > 0) {
            let temp = stack2.pop();
            yield temp.value;
        }
    }
}