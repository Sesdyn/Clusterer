# Clusterer

To build the documentation website:

```bash
make -C docs clean && make -C docs html && ghp-import -n -p docs/build/html --force
```