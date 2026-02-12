// Colors matching ChimeraX render.py
const MUTABLE_COLOR = 0xe05565;    // Coral-red for mutable regions
const IMMUTABLE_COLOR = 0x5078d0;  // Blue for immutable regions

export const MolstarViewer = ({
  cifUrl,
  immutableIndices = [],
  height = "600px",
  backgroundColor = "white",
  title = "Protein Structure"
}) => {
  const containerRef = React.useRef(null);
  const pluginRef = React.useRef(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    let mounted = true;
    const immutableSet = new Set(immutableIndices);

    const loadMolstar = async () => {
      if (!window.molstar) {
        await new Promise((resolve, reject) => {
          if (document.getElementById('molstar-script')) {
            const checkMolstar = setInterval(() => {
              if (window.molstar) {
                clearInterval(checkMolstar);
                resolve();
              }
            }, 100);
            setTimeout(() => {
              clearInterval(checkMolstar);
              reject(new Error('Timeout waiting for Mol* to load'));
            }, 10000);
            return;
          }

          const script = document.createElement('script');
          script.id = 'molstar-script';
          script.src = 'https://molstar.org/viewer/molstar.js';
          script.onload = () => setTimeout(resolve, 500);
          script.onerror = () => reject(new Error('Failed to load Mol* script'));
          document.head.appendChild(script);

          const link = document.createElement('link');
          link.rel = 'stylesheet';
          link.href = 'https://molstar.org/viewer/molstar.css';
          document.head.appendChild(link);
        });
      }
      return window.molstar;
    };

    const initViewer = async () => {
      try {
        const molstar = await loadMolstar();
        if (!mounted || !containerRef.current) return;

        if (!molstar?.Viewer) {
          throw new Error('Mol* Viewer not available');
        }

        // Create viewer with minimal UI
        const viewer = await molstar.Viewer.create(containerRef.current, {
          layoutIsExpanded: false,
          layoutShowControls: false,
          layoutShowRemoteState: false,
          layoutShowSequence: false,
          layoutShowLog: false,
          layoutShowLeftPanel: false,
          viewportShowExpand: false,
          viewportShowSelectionMode: false,
          viewportShowAnimation: false,
          collapseLeftPanel: true,
          collapseRightPanel: true,
        });

        pluginRef.current = viewer;
        const plugin = viewer.plugin;

        // Apply canvas settings for flat lighting and outlines
        if (plugin.canvas3d) {
          plugin.canvas3d.setProps({
            renderer: {
              backgroundColor: backgroundColor === 'white' ? 0xFFFFFF : 0x000000,
            },
            postprocessing: {
              outline: {
                name: 'on',
                params: {
                  scale: 2,
                  threshold: 0.33,
                  color: 0x000000,
                  includeTransparent: true,
                }
              },
              occlusion: { name: 'off', params: {} },
              shadow: { name: 'off', params: {} },
            }
          });
        }

        // Register custom color theme before loading structure
        registerCustomColorTheme(plugin, molstar, immutableSet);

        // Load structure with cartoon representation
        await viewer.loadStructureFromUrl(cifUrl, 'mmcif', false);

        // Wait for structure to be ready
        await new Promise(resolve => setTimeout(resolve, 200));

        // Apply custom color theme to all representations
        await applyCustomColoring(plugin);

        if (mounted) setLoading(false);
      } catch (err) {
        console.error('Mol* initialization error:', err);
        if (mounted) {
          setError(err.message);
          setLoading(false);
        }
      }
    };

    // Register a custom color theme that colors by mutable/immutable
    const registerCustomColorTheme = (plugin, molstar, immutableSet) => {
      try {
        // Access Color utility
        const Color = molstar.lib?.mol_util?.color?.Color;
        if (!Color) {
          console.warn('Color module not available');
          return;
        }

        // Create the custom theme provider
        const MutableImmutableColorThemeProvider = {
          name: 'mutable-immutable-coloring',
          label: 'Mutable/Immutable Coloring',
          category: 'Custom',
          factory: (ctx, props) => {
            // Get structure properties accessor
            const StructureProperties = molstar.lib?.mol_model?.structure?.StructureProperties;

            return {
              factory: MutableImmutableColorThemeProvider,
              granularity: 'group',
              preferSmoothing: true,
              color: (location) => {
                try {
                  if (StructureProperties?.residue?.label_seq_id) {
                    const seqId = StructureProperties.residue.label_seq_id(location);
                    if (immutableSet.has(seqId)) {
                      return Color(IMMUTABLE_COLOR);
                    }
                  }
                } catch (e) {
                  // Fallback to mutable color on error
                }
                return Color(MUTABLE_COLOR);
              },
              props: props,
              description: 'Colors residues based on mutable (coral-red) vs immutable (blue) regions'
            };
          },
          getParams: () => ({}),
          defaultValues: {},
          isApplicable: () => true
        };

        // Try to register the theme
        const registry = plugin.representation?.structure?.themes?.colorThemeRegistry;
        if (registry && !registry.has(MutableImmutableColorThemeProvider.name)) {
          registry.add(MutableImmutableColorThemeProvider);
        }
      } catch (e) {
        console.warn('Could not register custom color theme:', e);
      }
    };

    // Apply custom coloring to all structure representations
    const applyCustomColoring = async (plugin) => {
      try {
        const structures = plugin.managers.structure.hierarchy.current.structures;
        if (!structures || structures.length === 0) return;

        await plugin.dataTransaction(async () => {
          for (const structure of structures) {
            // Try to update representations with custom theme
            try {
              await plugin.managers.structure.component.updateRepresentationsTheme(
                structure.components,
                { color: 'mutable-immutable-coloring' }
              );
            } catch (e) {
              console.warn('Could not apply custom theme, trying uniform color:', e);
              // Fallback: try to apply uniform mutable color
              try {
                await plugin.managers.structure.component.updateRepresentationsTheme(
                  structure.components,
                  { color: 'uniform', colorParams: { value: MUTABLE_COLOR } }
                );
              } catch (e2) {
                console.warn('Uniform color also failed:', e2);
              }
            }
          }
        });
      } catch (e) {
        console.warn('Could not apply custom coloring:', e);
      }
    };

    initViewer();

    return () => {
      mounted = false;
      if (pluginRef.current) {
        try {
          pluginRef.current.dispose();
        } catch (e) {
          console.warn('Error disposing Mol* viewer:', e);
        }
      }
    };
  }, [cifUrl, backgroundColor, immutableIndices]);

  return (
    <div style={{ width: '100%', marginBottom: '2rem' }}>
      {title && <h3 style={{ marginBottom: '1rem' }}>{title}</h3>}
      <div style={{ position: 'relative', width: '100%', height }}>
        {loading && !error && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 10,
            textAlign: 'center',
            color: '#666',
          }}>
            <div>Loading protein structure...</div>
          </div>
        )}
        {error && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 10,
            textAlign: 'center',
            color: 'red',
            padding: '1rem',
          }}>
            <div>Error: {error}</div>
            <div style={{ fontSize: '0.8rem', marginTop: '0.5rem', color: '#666' }}>
              Check browser console for details
            </div>
          </div>
        )}
        <div
          ref={containerRef}
          style={{
            width: '100%',
            height: '100%',
            border: '1px solid #ddd',
            borderRadius: '8px',
            overflow: 'hidden',
            backgroundColor: backgroundColor,
          }}
        />
      </div>
    </div>
  );
};
