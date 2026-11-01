// GoldenEye Detector (Evaluation Edition) - small UI helpers

document.addEventListener("DOMContentLoaded", function () {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const fileNameLabel = document.getElementById("fileNameLabel");
  const panelHasil = document.getElementById("panel-hasil");

  if (dropzone && fileInput) {
    ["dragenter", "dragover"].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });
    dropzone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length) {
        fileInput.files = files;
        updateFileName();
      }
    });
    fileInput.addEventListener("change", updateFileName);
  }

  function updateFileName() {
    if (fileInput.files && fileInput.files.length && fileNameLabel) {
      fileNameLabel.textContent = "📄 " + fileInput.files[0].name;
    }
  }

  const flowModalEl = document.getElementById("flowDetailModal");
  if (flowModalEl) {
    flowModalEl.addEventListener("show.bs.modal", function (event) {
      const button = event.relatedTarget;
      const flowId = button.getAttribute("data-flow-id");
      const body = flowModalEl.querySelector(".modal-body");
      body.innerHTML =
        '<div class="text-center py-5"><div class="spinner-border text-primary"></div></div>';

      fetch(`/api/flow/${flowId}`)
        .then((res) => res.text())
        .then((html) => { body.innerHTML = html; })
        .catch(() => {
          body.innerHTML = '<div class="alert alert-danger m-3">Gagal memuat detail flow.</div>';
        });
    });
  }

  // ---- Panel hasil evaluasi: filter, pintasan, pagination tanpa reload ----
  function gantiPanel(url) {
    fetch(url)
      .then((res) => res.text())
      .then((html) => {
        const parser = new DOMParser();
        const dokumenBaru = parser.parseFromString(html, "text/html");
        const panelBaru = dokumenBaru.getElementById("panel-hasil");
        if (panelBaru) {
          panelHasil.innerHTML = panelBaru.innerHTML;
        }
      })
      .catch(() => {
        console.error("Gagal memuat data.");
      });
  }

  if (panelHasil) {
    panelHasil.addEventListener("click", function (e) {
      const link = e.target.closest(".ajax-link");
      if (link && !link.closest(".disabled")) {
        e.preventDefault();
        gantiPanel(link.getAttribute("href"));
      }
    });

    panelHasil.addEventListener("submit", function (e) {
      if (e.target.id === "filterForm") {
        e.preventDefault();
        const form = e.target;
        const params = new URLSearchParams(new FormData(form));
        gantiPanel(form.action + "?" + params.toString());
      }
    });
  }
});