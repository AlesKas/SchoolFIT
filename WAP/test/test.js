import { assert } from 'chai';
import { Tree } from "../tree.mjs";

let t = new Tree((a,b) => a < b);
let nums = [2,1,3];
nums.forEach(num => t.insertValue(num));

describe("preorder", function() {
    let preOrder = t.preorder(); 
    it ('root should be 2', function() {
        assert.equal(preOrder.next().value, 2);
    });
    it ('left child should be 1', function() {
        assert.equal(preOrder.next().value, 1);
    });
    it ('right child should be 3', function() {
        assert.equal(preOrder.next().value, 3);
    });
    it ('iterator should be finished', function() {
        assert.isTrue(preOrder.next().done);
    });
});

describe("inorder", function() {
    let inOrder = t.inorder();
    it ('left child should be 1', function() {
        assert.equal(inOrder.next().value, 1);
    });
    it ('root should be 2', function() {
        assert.equal(inOrder.next().value, 2);
    });
    it ('right child should be 3', function() {
        assert.equal(inOrder.next().value, 3);
    });
    it ('iterator should be finished', function() {
        assert.isTrue(inOrder.next().done);
    });
});

describe("postorder", function() {
    let postOrder = t.postorder();
    it ('left child should be 1', function() {
        assert.equal(postOrder.next().value, 1);
    });
    it ('right child should be 3', function() {
        assert.equal(postOrder.next().value, 3);
    });
    it ('root should be 2', function() {
        assert.equal(postOrder.next().value, 2);
    });
    it ('iterator should be finished', function() {
        assert.isTrue(postOrder.next().done);
    });
});

describe("2 iterators preorder", function() {
    let it1 = t.preorder();
    let it2 = t.preorder();
    it ('first iterator should be 2', function() {
        assert.equal(it1.next().value, 2);
    });
    it ('second iterator should be 2', function() {
        assert.equal(it2.next().value, 2);
    });
});

describe("2 iterators inorder", function() {
    let it1 = t.inorder();
    let it2 = t.inorder();
    it ('first iterator should be 1', function() {
        assert.equal(it1.next().value, 1);
    });
    it ('second iterator should be 1', function() {
        assert.equal(it2.next().value, 1);
    });
});

describe("2 iterators postorder", function() {
    let it1 = t.inorder();
    let it2 = t.inorder();
    it ('first iterator should be 1', function() {
        assert.equal(it1.next().value, 1);
    });
    it ('second iterator should be 1', function() {
        assert.equal(it2.next().value, 1);
    });
});