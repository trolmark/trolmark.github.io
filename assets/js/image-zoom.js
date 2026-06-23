document.addEventListener("click", (event) => {
  const source = event.target.closest(".page-image-grid img");

  if (!source) {
    return;
  }

  const dialog = document.createElement("dialog");
  dialog.className = "image-zoom-dialog";

  const closeButton = document.createElement("button");
  closeButton.type = "button";
  closeButton.className = "image-zoom-close";
  closeButton.setAttribute("aria-label", "Close enlarged image");
  closeButton.innerHTML = "&times;";

  const image = document.createElement("img");
  image.src = source.currentSrc || source.src;
  image.alt = source.alt;

  closeButton.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (dialogEvent) => {
    if (dialogEvent.target === dialog) {
      dialog.close();
    }
  });
  dialog.addEventListener("close", () => dialog.remove());

  dialog.append(closeButton, image);
  document.body.append(dialog);
  dialog.showModal();
});
