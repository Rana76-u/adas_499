import 'package:flutter/material.dart';
import 'package:adas_499/Core/model_config.dart'
    show ModelConfig, availableModels;

class CustomAppbar extends StatelessWidget implements PreferredSizeWidget{
  final bool _modelLoaded;
  final ModelConfig? _selectedModel;
  final ValueChanged<ModelConfig>? onModelChanged;

  const CustomAppbar({
    super.key,
    required bool modelLoaded,
    required ModelConfig? selectedModel,
    this.onModelChanged,
  }) : _modelLoaded = modelLoaded,
       _selectedModel = selectedModel;

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);

  @override
  Widget build(BuildContext context) {
    return AppBar(
      toolbarHeight: kToolbarHeight,
        backgroundColor: const Color(0xFF0D0D1F),
        elevation: 0,
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: const Color(0xFF1A73E8),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(
                Icons.center_focus_strong,
                color: Colors.white,
                size: 20,
              ),
            ),
            const SizedBox(width: 10),
            const Text(
              'DriveAid',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w700,
                color: Colors.white,
              ),
            ),
          ],
        ),
        actions: [
          if (_selectedModel != null)
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<ModelConfig>(
                  value: _selectedModel,
                  dropdownColor: const Color(0xFF0D0D1F),
                  style: const TextStyle(color: Colors.white, fontSize: 12),
                  iconEnabledColor: Colors.white70,
                  isDense: true,
                  items: availableModels.map((model) {
                    return DropdownMenuItem(
                      value: model,
                      child: Text(
                        model.name,
                        style: TextStyle(
                          color: _modelLoaded ? Colors.white : Colors.white54,
                          fontSize: 12,
                        ),
                      ),
                    );
                  }).toList(),
                  onChanged: _modelLoaded && onModelChanged != null
                      ? (newModel) {
                          if (newModel != null) {
                            onModelChanged!(newModel);
                          }
                        }
                      : null,
                ),
              ),
            ),
          Container(
            margin: const EdgeInsets.only(right: 12),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: _modelLoaded
                  ? Colors.green.withValues(alpha: 0.2)
                  : Colors.red.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: _modelLoaded
                    ? Colors.greenAccent.withValues(alpha: 0.6)
                    : Colors.redAccent.withValues(alpha: 0.6),
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 6,
                  height: 6,
                  decoration: BoxDecoration(
                    color: _modelLoaded ? Colors.greenAccent : Colors.redAccent,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 5),
                Text(
                  _modelLoaded ? 'Active' : 'Loading',
                  style: TextStyle(
                    color: _modelLoaded ? Colors.greenAccent : Colors.redAccent,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ],
      );
  }
}