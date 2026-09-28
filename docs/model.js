// Mock model until export is done
function score(input) {
    // Return an array of scores for each class
    // In our case we have 26 letters + space + del + nothing
    return new Array(29).fill(0.0);
}
