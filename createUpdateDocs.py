import os
import logging

log = logging.getLogger(os.path.basename(__file__))


def moduleTemplate(modulePath: str):
    moduleName = os.path.basename(modulePath).replace("_", r'\_')
    moduleName = moduleName[:-3]
    modulePath = modulePath[:-3]
    template = \
        f"""{moduleName}
{"="*len(moduleName)}

.. automodule:: {modulePath.replace(os.path.sep, ".")}
   :members:
   :undoc-members:

.. toctree::
   :glob:

   *
"""
    return template


def packageTemplate(packagePath: str):
    packageName = os.path.basename(packagePath).replace("_", r'\_')
    template = \
        f"""{packageName}
{"="*len(packageName)}

.. automodule:: {packagePath.replace(os.path.sep, ".")}
   :members:
   :undoc-members:

.. toctree::
   :glob:

   {packageName}/*
"""
    return template


def makeDocu(basedir=os.path.join("src/sensai"), overwrite=False):
    """
    Creates/updates documentary if form of rst files for modules and packages. It should be executed from the top

    :param basedir: path to library basedir, typically "src/<library_name>"
    :param overwrite: if True, will overwrite existing rst files
    :return:
    """
    libraryBasedir = basedir.split(os.path.sep, 1)[1]  # splitting off the "src" part
    for file in os.listdir(basedir):
        if file.startswith("_"):
            continue

        libraryFilePath = os.path.join(libraryBasedir, file)
        fullPath = os.path.join(basedir, file)
        fileName, ext = os.path.splitext(file)
        docsFilePath = os.path.join("docs", libraryBasedir, f"{fileName}.rst")
        if os.path.exists(docsFilePath) and not overwrite:
            log.debug(f"{docsFilePath} already exists, skipping it")
            if os.path.isdir(fullPath):
                makeDocu(basedir=fullPath, overwrite=overwrite)
            continue
        os.makedirs(os.path.dirname(docsFilePath), exist_ok=True)

        if ext == ".py":
            log.info(f"writing module docu to {docsFilePath}")
            with open(docsFilePath, "w") as f:
                f.write(moduleTemplate(libraryFilePath))
        elif os.path.isdir(fullPath):
            log.info(f"writing package docu to {docsFilePath}")
            with open(docsFilePath, "w") as f:
                f.write(packageTemplate(libraryFilePath))
            makeDocu(basedir=fullPath, overwrite=overwrite)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    makeDocu()
