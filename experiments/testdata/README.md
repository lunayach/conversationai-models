# Test data

* "Optional" is when a label may or may not be present, but when it is not, it
  should be treated as having value 0.
* "Partial" is when label may or may not be present, and when it is, it
  specifies it's value. When it is not, nothing can be said about the value
  (models should not back-prop through the label if the label is not present).
