//
//  NsfwViewController.swift
//  ImageClassification
//
//  Created by dave on 2020/2/9.
//  Copyright © 2020 Y Media Labs. All rights reserved.
//

import Foundation

import AVFoundation
import UIKit

class NsfwViewController: UIViewController,UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    @IBOutlet var imageView: UIImageView!
    @IBOutlet var imageView1: UIImageView!
    @IBOutlet var chooseBuuton: UIButton!
    @IBOutlet var resLablel: UILabel!
    
    var imagePicker = UIImagePickerController()
    // Handles all data preprocessing and makes calls to run inference through the `Interpreter`.
    private var modelDataHandler: NsfwModelDataHandler? = NsfwModelDataHandler(modelFileInfo: NsfwMobileNet.modelInfo)
    // MARK: View Handling Methods
    override func viewDidLoad() {
      super.viewDidLoad()

//      guard modelDataHandler != nil else {
//        fatalError("Model set up failed")
//      }
//
//      cameraCapture.delegate = self
//
//      addPanGesture()
    }
    @IBAction func btnClicked() {

        if UIImagePickerController.isSourceTypeAvailable(.savedPhotosAlbum){
            print("Button capture")
            imagePicker.delegate = self
            imagePicker.sourceType = .savedPhotosAlbum
            imagePicker.allowsEditing = false

            present(imagePicker, animated: true, completion: nil)
        }
    }
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
         picker.dismiss(animated: true, completion: nil)
         guard let image = info[.originalImage] as? UIImage else {
             fatalError("Expected a dictionary containing an image, but was provided the following: \(info)")
         }
        
         print("image w:\(image.size.width)h:\(image.size.height)")
         imageView.image = image
         let data = image.scaledData(with: CGSize(width:256, height: 256))
//         let manager = FileManager.default
//         let urlForDocument = manager.urls(for: .documentDirectory, in:.userDomainMask)
//         let url = urlForDocument[0] as URL
           let filePath:String = NSHomeDirectory() + "/Documents/bb_data.txt"
           print(filePath)
           do{
            try data!.write(to: URL.init(fileURLWithPath: filePath))
           }catch {
            print(error)
           }
         imageView1.image = UIImage.init(data:  image.scaledData(with: CGSize(width:256, height: 256))!)
         modelDataHandler?.classify(image: image
            , completion: { result1 in
                switch result1 {
                           case let .success(classificationResult):
                               //print(classificationResult.nsfw)
                              self.resLablel.text = "\(classificationResult.nsfw)"
                              if(classificationResult.nsfw>0.3){
                                self.resLablel.textColor = UIColor.red
                              }else{
                                self.resLablel.textColor = UIColor.green
                              }
                              print("nsfw nsfw:\(classificationResult.nsfw) sfw:\(classificationResult.sfw)")
                           case .error(_):
                               print("is error")
                           }
        })
        
          
        // pickImageCallback?(image)
     }
    

}
